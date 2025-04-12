import numpy as np
import os
import torchvision
from torch.utils.data import Dataset
import torch
import torchaudio
from itertools import permutations
import clip
import utils as u
from beats.BEATs import BEATs, BEATsConfig
import json
import transformers
from pathlib import Path
from face_detector.detect_face import FaceDetector
from paddleocr import PaddleOCR
import pkg_resources
from symspellpy.symspellpy import SymSpell

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SINGULAR_FEATURES = ('asr_sentiment', 'ocr_sentiment')

transformers.logging.set_verbosity_error()

def get_device(model):
    return next(model.parameters()).device.type


class FaceExtractAndClassify(torch.nn.Module):
    # Extracts faces and then classifies their emotions
    # Sources:
    # https://github.com/ultralytics/yolov5
    # https://huggingface.co/trpakov/vit-face-expression
    def __init__(self, use_layernorm=False):
        super(FaceExtractAndClassify, self).__init__()
        self.face_detector_model = FaceDetector()
        self.face_emotion_model = FaceEmotionClassifier(use_layernorm=use_layernorm)

    def to_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available.')
        elif device not in ('cpu', 'cuda'):
            print('Device can only be cpu or cuda.')
        else:
            self.to(device)

    def get_device(self):
        return get_device(self)

    def process_video(self, input_tensor=None, video_path=None, fps=None, use_scenecuts=False, n_frames=None, **kwargs):
        if input_tensor is None and video_path == None:
            raise ValueError('You should provide either input frames or video path.')
        elif input_tensor is None:
            if use_scenecuts:
                input_tensor, _ = u.video_to_midscenes(video_path)  # Get scenes  
            else:
                input_tensor, input_fps = u.extract_frames(str(video_path), output_fps=fps)
            
            if n_frames != None:
                indices = u.equidistant_indices(input_tensor.shape[0], n_frames)
                input_tensor = input_tensor[indices, ...]

        video_features = []
        video_predictions = []
        video_coordinates = []
        for input_frame in input_tensor:
            input_frame = input_frame.squeeze()
            faces, coordinates = self.face_detector_model(input_frame)
            video_coordinates.append(coordinates)
            if faces == []:
                frame_predictions, frame_features = None, None
            else:
                face_sizes = [face.size for face in faces]
                largest_face_inds = np.argsort(np.array(face_sizes))[::-1]
                # get two largest faces
                frame_features = []
                frame_predictions = []
                for face in faces:
                    face_output = self.face_emotion_model(face)
                    frame_features.append(face_output['features'].detach())
                    frame_predictions.append(face_output['predictions'])

                two_largest_inds = largest_face_inds[:2].tolist()

                # get mean of two largest faces
                frame_features = [frame_features[i] for i in two_largest_inds]
                frame_features = torch.stack(frame_features, 0).mean(0)
                video_features.append(frame_features)
            video_predictions.append(frame_predictions)
                
        if video_features != []:
            video_features = torch.stack(video_features, 0)

        return {'features': video_features, 'predictions': video_predictions, 'coordinates': video_coordinates}

class FaceEmotionClassifier(torch.nn.Module):
    # Classifies the emotion of a face
    # Source: https://huggingface.co/trpakov/vit-face-expression
    def __init__(self, use_layernorm=False):
        super(FaceEmotionClassifier, self).__init__()
        model_name = "trpakov/vit-face-expression"
        self.model = transformers.AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(
                model_name, _from_pipeline='image_classification')
        self.to_tensor = torchvision.transforms.ToTensor()
        self.layernorm = torch.nn.LayerNorm(768, eps=1e-12).to(DEVICE) if use_layernorm else None

    def get_device(self):
        return get_device(self)

    def __call__(self, pil_image):
        with torch.no_grad():
            image = self.image_processor(pil_image)
            image = torch.Tensor(image['pixel_values'][0]).unsqueeze(0).to(DEVICE)
            output = self.model(image, output_hidden_states=True)
            features = output.hidden_states[-1]     # last hidden state
            if self.layernorm != None:
                features = self.layernorm(features)
            # Output corresponding to the <CLS> token
            features = features[:, 0, :].squeeze().detach()
            probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
            max_val, max_ind = torch.max(probabilities, dim=-1)
            max_val = max_val.item()
            max_ind = max_ind.item()
            predicted_label = self.id2label[max_ind]
            prediction = (predicted_label, max_val)
            return {'features': features, 'predictions': prediction}
        
    def sanity_check(self):
        import requests
        from PIL import Image
        import matplotlib.pyplot as plt
        import numpy as np

        url = "https://img.freepik.com/premium-photo/3d-rendered-illustration-angry-man-face_181203-19310.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        output = self.__call__(image)
        image = np.array(image)
        plt.figure()
        plt.imshow(image)
        plt.title(f"{output['predictions'][0]}: {output['predictions'][1]:.2f}")
        plt.savefig('face_emotion.png')



class FastASR(torch.nn.Module):
    # Automatic speech recognition, plus translator into English
    # Source: https://huggingface.co/openai/whisper-tiny
    # Chunk length should be smaller than 30
    def __init__(self, no_cuda=False, amp=False, chunk_length_s=30, batch_size=16, use_tiny_model=False):
        super(FastASR, self).__init__()
        self.amp = amp

        if chunk_length_s > 30:
            Warning('Chunk length longer than 30 seconds can result in errors.')
        if torch.cuda.is_available() and not no_cuda:
            dtype = torch.float16
        else:
            dtype = torch.float32

        if use_tiny_model:
            model_id = "openai/whisper-tiny"
        else:
            model_id = "openai/whisper-large-v3"

        model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True)
        model.eval()

        processor = transformers.AutoProcessor.from_pretrained(model_id)

        self.pipe = transformers.pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=True,
            return_language=True,
            torch_dtype=dtype,
            device=DEVICE,
        )

    def get_device(self):
        return self.pipe.model.device.type
    
    def process_video(self, input_audio=None, video_path=None, sr=None):
        if input_audio is None and video_path == None:
            return {'text': '', 'chunks': []}
        elif input_audio is None:
            input_audio, sr = u.extract_audio(video_path)
            if input_audio is None:
                return {'text': '', 'chunks': []}
        elif sr == None:
            raise ValueError('If you are providing input audio, you should also provide sampling rate.')
        
        input_audio = input_audio.mean(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        input_audio = resampler(torch.Tensor(input_audio)).numpy()

        # Use translation by default. The model works at the same speed.
        # No problem if the source is in English.

        output = self.pipe(input_audio, generate_kwargs={"task": "translate"})
        return output


class ASRSentiment(torch.nn.Module):
    # Wrapper for automatic speech recognition, followed by sentiment analysis on its output text.
    def __init__(self, tiny_asr=False):
        super(ASRSentiment, self).__init__()
        self.asr_model = FastASR(use_tiny_model=tiny_asr)
        self.sentiment_model = SentimentClassifier()
        self.to_device('cpu')

    def to_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available.')
        elif device not in ('cpu', 'cuda'):
            print('Device can only be cpu or cuda.')
        else:
            self.asr_model.pipe.model.to(device)
            self.sentiment_model.to(device)

    def process_video(self, input_tensor=None, video_path=None, sr=None, **kwargs):
        output = self.asr_model.process_video(input_tensor, video_path, sr=sr)
        text = output['text'].strip()

        # If there is only one language, use it
        languages = list(dict.fromkeys([chunk['language'] for chunk in output['chunks']]))
        if len(languages) == 1:
            language = languages[0]
        else:
            language = None
        if text == '':
            output = {'features': [], 'predictions': [], 'language': None}
        else:
            output = self.sentiment_model(text)
        output['asr'] = text
        output['language'] = language
        return output

class SentimentClassifier(torch.nn.Module):
    # Classifies sentiment of a given text.
    # Source: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
        self.config = transformers.AutoConfig.from_pretrained(MODEL)
        self.config.output_hidden_states = True
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL, config=self.config)
        self.model.eval()

    def get_device(self):
        return get_device(self)

    def __call__(self, text):
        with torch.no_grad():
            x = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            x = x.to(self.model.device)
            y = self.model(**x)
            # Get features
            features = y.hidden_states[-1][:, 0, :]
            # Get predictions
            # Logits are created w.r.t. position 0 (<CLS>)
            logits = torch.nn.functional.softmax(y.logits, dim=-1)
            max_val, max_ind = torch.max(logits, dim=-1)
            predictions = (self.config.id2label[max_ind.item()], max_val.item())
            return {"features": features, "predictions": predictions}
    
    def process_video(self, text):
        if text == '':
            output = {'features': [], 'prediction': []}
        else:
            output = self.__call__('. '.join(text))
        return output
    

def generate_caption_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):
    # Generator function used in CLIP caption generation
    # Source: https://github.com/rmokady/CLIP_prefix_caption
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


class CaptionMLP(torch.nn.Module):
    # MLP model used in CLIP caption model
    # Source: https://github.com/rmokady/CLIP_prefix_caption

    def __init__(self, sizes, bias=True, act=torch.nn.Tanh):
        super(CaptionMLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def get_device(self):
        return get_device(self)

    
class ClipCaptionModel(torch.nn.Module):
    # Generates caption given CLIP features
    # Source: https://github.com/rmokady/CLIP_prefix_caption
    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = CaptionMLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))

    def to_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available.')
        elif device not in ('cpu', 'cuda'):
            print('Device can only be cpu or cuda.')
        else:
            self.to(device)

    def get_dummy_token(self, batch_size, device):
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens, prefix, mask = None, labels = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    
class CLIPRunner(torch.nn.Module):
    # Extracts visual features
    # CLIP: Contrastive Language-Image Pretraining
    # Source: https://github.com/openai/CLIP
    def __init__(self, model='ViT-B/32', no_cuda=False):
        super(CLIPRunner, self).__init__()

        self.model, self.preprocessor = clip.load(model, device=torch.device("cpu"), jit=False)
        self.model.eval()
        self.preprocessor = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)), 
                torchvision.transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)), 
                torchvision.transforms.Lambda(lambda x: x / 255),
                torchvision.transforms.Resize(
                    size=224, 
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC, 
                    max_size=None,
                    antialias=True),
                torchvision.transforms.CenterCrop(size=(224, 224)),
                torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ]
        )
        # Only vision transformer is needed
        del self.model.ln_final, self.model.logit_scale, self.model.positional_embedding, \
            self.model.text_projection, self.model.token_embedding, self.model.transformer
        
    def get_device(self):
        return get_device(self)
    
    def to_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available.')
        elif device not in ('cpu', 'cuda'):
            print('Device can only be cpu or cuda.')
        else:
            self.to(device)

    def encode(self, image):
        # Gets image encoding using CLIP
        with torch.no_grad(): 
            if len(image.shape) == 3:   # add batch dimension
                image = image.unsqueeze(0)

            image_features = self.model.encode_image(image)
            top_predictions = None

        return {"predictions": top_predictions, "features": image_features}
    
    def get_preprocessor(self):
        return self.preprocessor
    
    def process_video(self, input_tensor=None, video_path=None, batch_size=8, use_scenecuts=False, n_frames=None, fps=1, **kwargs):
        if input_tensor is None and video_path == None:
            raise ValueError('You should provide either input frames or video path.')
        elif input_tensor is None:
            if use_scenecuts:
                input_tensor, _ = u.video_to_midscenes(video_path)  # Get scenes  
            else:
                input_tensor, input_fps = u.extract_frames(str(video_path), output_fps=fps)  
            
            if n_frames != None:
                indices = u.equidistant_indices(input_tensor.shape[0], n_frames)
                input_tensor = input_tensor[indices, ...]

        input_tensor = self.preprocessor(input_tensor)

        dataset = MyTensorDataset(input_tensor, preprocessor=None)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
        video_output = {'predictions': [], 'features': []}
        
        device = self.get_device()
        for batch in dataloader: 
            batch = batch.to(device)
            batch_output = self.encode(batch)
            video_output['features'].append(batch_output['features'])
            video_output['predictions'] += [batch_output['predictions']]
        video_output['features'] = torch.cat(video_output['features'], dim=0).detach()
        
        return video_output
    

class CaptionRunner(torch.nn.Module):
    # Wrapper for caption generator
    def __init__(self):
        super(CaptionRunner, self).__init__()
        self.caption_prefix_length = 10
        self.caption_model = ClipCaptionModel(self.caption_prefix_length)
        model_path = Path('weights/conceptual_weights.pt')
        if not model_path.exists():
            u.download_gdrive('14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT', model_path)
        self.caption_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False) 
        self.caption_model.eval()
        self.caption_tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

    def to_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available.')
        elif device not in ('cpu', 'cuda'):
            print('Device can only be cpu or cuda.')
        else:
            self.caption_model.to_device(device)

    def __call__(self, features):
        device = get_device(self)
        features = torch.Tensor(features).to(device)
        embedded = self.caption_model.clip_project(features)
        embedded = embedded.reshape(features.shape[0], self.caption_prefix_length, -1)
        embedded = embedded.mean(0, keepdims=True)      # Get prediction for the mean of the frames
        caption = generate_caption_beam(self.caption_model, self.caption_tokenizer, embed=embedded)[0]
        return caption

class BEATSRunner(torch.nn.Module):
    # Audio classifier
    # Source: https://github.com/microsoft/unilm/tree/master/beats
    def __init__(self, predict=True):
        super(BEATSRunner, self).__init__()
        self.predict = predict
        script_dir = os.path.dirname(__file__)
        script_dir = Path(script_dir)
        with open(script_dir / "beats/pretrained_models_labels/ontology.json", "r") as f:
            labels = json.load(f)
        self.id_to_label = {label['id']: label['name'] for label in labels}
        self.target_sample_rate = 16000

        # load the pre-trained checkpoints
        model_path = script_dir / 'beats/pretrained_models_labels/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
        if not model_path.exists():
            url = "https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
            u.download(url, model_path)

        checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
        
        self.label_dict = checkpoint["label_dict"]
        cfg = BEATsConfig(checkpoint['cfg'])
        self.BEATs_model = BEATs(cfg)
        self.BEATs_model.load_state_dict(checkpoint['model'])
        self.BEATs_model.eval()

        if not self.predict:
            self.BEATs_model.predictor = None
    
    def __call__(self, input_):
        with torch.no_grad():
            padding_mask = torch.zeros(input_.shape).bool().to(DEVICE) 
            output = self.BEATs_model.extract_features(input_, padding_mask=padding_mask)
            output = output.mean(1)
            return output

    def to_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available.')
        elif device not in ('cpu', 'cuda'):
            print('Device can only be cpu or cuda.')
        else:
            self.to(device)

        
    def process_video(self, input_tensor=None, video_path=None, n_frames=None, batch_size=8, num_workers=2, fps=None, overlap=0.5, sr=None, **kwargs):
        video_output = {'features': [], 'predictions': []}
        if input_tensor is None and video_path == None:
            return video_output
        elif input_tensor is None:
            input_tensor, sr = u.extract_audio(video_path)
            if input_tensor is None:
                return {'features': [], 'predictions': []}
            input_tensor = torch.Tensor(input_tensor)
        elif sr == None:
            raise ValueError('If you are providing input audio, you should also provide sampling rate.')

        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)

        if fps != None:
            n_frames = None
            step_size_s = 1 / fps
            chunk_length_s = step_size_s / overlap
        else:
            chunk_length_s = 3

        input_length_s = input_tensor.shape[-1] / sr
        if input_length_s < chunk_length_s:
            return {'features': [], 'predictions': []}

        loader = AudioDataset(input_tensor, sr, n_frames, chunk_length_s=chunk_length_s, overlap=overlap, sr_output=16000)
        loader = torch.utils.data.DataLoader(loader, batch_size=batch_size, num_workers=num_workers)
    
        for input_ in loader:
            input_ = input_.to(DEVICE)
            output = self.__call__(input_)
            video_output['features'].append(output)
        video_output['features'] = torch.cat(video_output['features'], dim=0)
        video_output['predictions'] = None

        if self.predict:
            mean_feature = video_output['features'].mean(0)
            logits = self.BEATs_model.predictor(mean_feature)
            logits = torch.sigmoid(logits)

            topk_values, topk_indices = torch.topk(logits, k=5, dim=-1)
            predictions = [(self.id_to_label[self.label_dict[idx.item()]], val.item()) 
                        for idx, val in zip(topk_indices, topk_values)]
            video_output['predictions'] = predictions

        return video_output
    

class AudioDataset(Dataset):
    # Loads audio in chunks
    def __init__(self, audio, sr_input, n_chunks=None, overlap=0.5, chunk_length_s=3, sr_output=None):
        audio = torch.Tensor(audio)
        resampler = torchaudio.transforms.Resample(sr_input, sr_output)
        audio = resampler(audio)  # resample
        if audio.size(0) >= 2:     # make mono
            audio = audio.mean(0)
        audio = audio.squeeze()

        # Get chunks of audio
        chunk_length_sample = int(round(chunk_length_s * sr_output))
        step_sample = int(round(sr_output * chunk_length_s * overlap))

        inds = np.round(np.arange(0, audio.shape[0] - chunk_length_sample, step_sample)).astype(np.int32).tolist()
        
        if n_chunks != None and len(inds) > n_chunks:
            inds = u.equidistant_indices(len(inds), n_chunks)

        self.audio = [audio[ind:ind+chunk_length_sample] for ind in inds]

        # Get timestamps
        seconds_tensor = [ind / sr_output for ind in inds]
        self.timestamps = []
        for seconds in seconds_tensor:
            minute, second = divmod(seconds, 60)
            minute = int(minute)
            second, milisecond = divmod(second, 1)
            second = int(second)
            millisecond = int(round(milisecond * 1000))
            self.timestamps.append(f"{minute:02d}:{second:02d}.{millisecond:03d}")       

    def __getitem__(self, index):
        return self.audio[index]

    def __len__(self):
        return len(self.audio)
    
    def get_timestamps(self):
        return self.timestamps
    

class MyTensorDataset(Dataset):
    # Creates a loader given a large tensor.
    def __init__(self, data, preprocessor=None):
        super().__init__()
        self.preprocessor = preprocessor
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.preprocessor != None:
            sample = self.preprocessor(sample)
        return sample


class ToEnglishTranslator(torch.nn.Module):
    # Translator into English
    # Source: https://huggingface.co/facebook/nllb-200-distilled-600M
    def __init__(self):
        super(ToEnglishTranslator, self).__init__()
        model_name = 'facebook/nllb-200-distilled-600M'
        self.translator = transformers.pipeline("translation", model=model_name, device=DEVICE)
        self.translator.model.eval()
        self.mappings_flores = {"arabic": 'arb_Arab', "bulgarian": 'bul_Cyrl', "german": 'deu_Latn', "greek": 'ell_Grek', "english": 'eng_Latn', "spanish": 'spa_Latn', "french": 'fra_Latn', "hindi": 'hin_Deva', "italian": 'ita_Latn', "japanese": 'jpn_Jpan', "dutch": 'nld_Latn', "polish": 'pol_Latn', "portuguese": 'por_Latn', "russian": 'rus_Cyrl', "swahili": 'swh_Latn', "thai": 'tha_Thai', "turkish": 'tur_Latn', "urdu": 'urd_Arab', "vietnamese": 'vie_Latn', "chinese": 'zho_Hant', "unknown": "unknown"}

    def __call__(self, text, language):
        if language not in ('english', 'unknown'):
            with torch.no_grad():
                return self.translator(text, src_lang=self.mappings_flores[language], tgt_lang='eng_Latn')[0]['translation_text']
        else:
            return text

class SpellChecker(torch.nn.Module):
    # Spellchecker, corrector
    # Source: https://huggingface.co/ai-forever/T5-large-spell
    def __init__(self):
        super(SpellChecker, self).__init__()
        path_to_model = "ai-forever/T5-large-spell"
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(path_to_model)
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path_to_model)
        self.prefix = "grammar: "

    def __call__(self, texts):
        with torch.no_grad():
            single_input = isinstance(texts, str)
            if single_input:
                texts = [texts]
            texts = [self.prefix + text for text in texts]
            encodings = self.tokenizer(texts, return_tensors="pt", padding=True)
            device = get_device(self)
            encodings = encodings.to(device)
            generated_tokens = self.model.generate(**encodings, max_length=512)
            output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            if single_input:
                output = output[0]
            return output


class TextLanguageClassifier(torch.nn.Module):
    # Predicts the language given text
    # This is used to create an Any-To-English translator
    # Source: https://huggingface.co/papluca/xlm-roberta-base-language-detection
    def __init__(self):
        super(TextLanguageClassifier, self).__init__()
        self.model = transformers.pipeline(
            "text-classification", 
            model="papluca/xlm-roberta-base-language-detection",
            device=DEVICE
            )
        self.model.model.eval()
        self.to_device('cpu')
        self.max_length = self.model.model.config.max_position_embeddings - 2
        self.languages = {"ar": "arabic", "bg": "bulgarian", "de": "german", "el": "greek", "en": "english", "es": "spanish", "fr": "french", "hi": "hindi", "it": "italian", "ja": "japanese", "nl": "dutch", "pl": "polish", "pt": "portuguese", "ru": "russian", "sw": "swahili", "th": "thai", "tr": "turkish", "ur": "urdu", "vi": "vietnamese", "zh": "chinese", "unknown": "unknown"}
        self.threshold = 0.9

    def get_device(self):
        return self.model.model.device.type

    def to_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available.')
        elif device not in ('cpu', 'cuda'):
            print('Device can only be cpu or cuda.')
        else:
            self.model.model.to(device)

    def __call__(self, texts):
        with torch.no_grad():
            single_input = isinstance(texts, str)
            if single_input:
                texts = [texts]
            output = self.model(texts, truncation=True, max_length=self.max_length)
            output = [self.languages[output_frame['label']] if output_frame['score'] > self.threshold else 'unknown' for output_frame in output]
            if single_input:
                output = output[0]
            return output


class OCRRunner:
    # Wrapper for optical Character Recognition (OCR) model to process full videos.
    # Source: https://github.com/PaddlePaddle/PaddleOCR
    def __init__(self, threshold=0.75):
        
        self.threshold = threshold
        self.model = PaddleOCR(
            show_log=False, 
            use_angle_cls=False, 
            lang='en', 
            use_gpu=torch.cuda.is_available(),
            )

    def get_device(self):
        return DEVICE

    def __call__(self, img_path, **kwargs):
        with torch.no_grad():
            output = self.model.ocr(img_path, cls=True)[0]
            if output == None:
                text = ''
            else:
                text = " ".join([line[1][0] for line in output if line[1][1] > self.threshold])
                text = text.lower()
                output = [box for box in output if box[1][1] > self.threshold]
            
            return text, output

    def process_video(self, input_frames=None, video_path=None, fps=None, use_scenecuts=False, n_frames=None):
        if input_frames is None and video_path == None:
            raise ValueError('You should provide either input frames or video path.')
        elif input_frames is None:
            if use_scenecuts:
                input_frames, _ = u.video_to_midscenes(video_path)  # Get scenes  
            else:
                input_frames, input_fps = u.extract_frames(str(video_path), output_fps=fps)
            
            if n_frames != None:
                indices = u.equidistant_indices(input_frames.shape[0], n_frames)
                input_frames = input_frames[indices, ...]

        video_texts = []
        video_outputs = []

        for frame in input_frames: 
            frame_text, frame_output = self.__call__(frame)
            video_texts.append(frame_text)
            video_outputs.append(frame_output)

        return video_texts, video_outputs
    

class OCRPipeline(torch.nn.Module):
    ''' Pipeline:
    Predict language
    
    If not English:
        Translate to English

    If English:
        Segment words (add space where necessary)
        Correct spelling

    Analyze sentiment
    '''
    def __init__(self, verbose=False):
        super(OCRPipeline, self).__init__()
        self.verbose = verbose
        self.ocr_model = OCRRunner()
        self.language_classifier = TextLanguageClassifier()
        self.segmentor = TextSegmentor()
        self.spellchecker = SpellChecker()
        self.translator = ToEnglishTranslator()
        self.sentiment_classifier = SentimentClassifier()

    def to_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available.')
        elif device not in ('cpu', 'cuda'):
            print('Device can only be cpu or cuda.')
        else:
            self.language_classifier.model.model.to(device)


    def process_video(self, input_tensor=None, video_path=None, fps=None, **kwargs):
        ocr_texts, ocr_outputs = self.ocr_model.process_video(input_tensor, video_path, fps)
        output = {}
        output['ocr_raw'] = ocr_texts
        output['boxes'] = ocr_outputs
        unique_texts = list(dict.fromkeys(ocr_texts))    # Take unique
        if '' in unique_texts:
            unique_texts.remove('')
        if unique_texts == []:
            output['features'] = []
            output['ocr_processed'] = []
        else:
            languages = self.language_classifier(unique_texts)           
            processed_texts = []
            for i in range(len(unique_texts)):
                if languages[i] in ('english', 'unknown'):
                    segmented = self.segmentor(unique_texts[i])
                    corrected = self.spellchecker(segmented)
                    processed_texts.append(corrected)
                else:
                    translated = self.translator(unique_texts[i], languages[i])
                    processed_texts.append(translated)

            processed_texts = list(dict.fromkeys(processed_texts))
            output['ocr_processed'] = processed_texts

            output_text = '. '.join(processed_texts)
            model_output = self.sentiment_classifier(output_text)
            output.update(model_output)
                 
        return output


class TextSegmentor:
    # Segments text into word.
    # Useful when OCR model misses the spaces.
    # Source: https://pypi.org/project/symspellpy/
    def __init__(self):
        self.model = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
        self.model.load_dictionary(dictionary_path, term_index=0, count_index=1)
    def __call__(self, text):
        if text in ('-', ''):
            return text
        else:
            output = self.model.word_segmentation(text).corrected_string
            return output


'''Emotion classifier:
Processes sequential pretrained features, namely
BEATs (audio classification), CLIP (image analysis),
and face emotion, using pairwise matched multi-head cross-attention.
Adds non-sequential features, namely sentiments of ASR and OCR.
Finally classifies the resulting vector into emotion.
'''

class AttentionClassifier(torch.nn.Module):
    def __init__(self, feature_dims, d_model, d_output, dropout, num_heads=4):
        super(AttentionClassifier, self).__init__()

        self.sequence_features = [feature for feature in feature_dims.keys() if feature not in SINGULAR_FEATURES]
        
        self.d_model = d_model
        self.num_heads = num_heads

        # Create encoders for each modality
        self.encoders = torch.nn.ModuleDict({feature: torch.nn.Linear(feature_dims[feature], d_model) for feature in feature_dims.keys()})

        # Start calculating the concatenated input dimension for the classification layer
        cls_input_dim = d_model * (len(feature_dims) - len(self.sequence_features))

        # Create linear layers for each modality
        self.query_proj = torch.nn.ModuleDict({feature: torch.nn.Linear(d_model, d_model) for feature in self.sequence_features})
        self.key_proj = torch.nn.ModuleDict({feature: torch.nn.Linear(d_model, d_model) for feature in self.sequence_features})
        self.value_proj = torch.nn.ModuleDict({feature: torch.nn.Linear(d_model, d_model) for feature in self.sequence_features})
        # Create multi-head attention layers for each permutation of modality pairs
        self.attention_layers = torch.nn.ModuleDict()
        self.attn_norms = torch.nn.ModuleDict()
        for (feat1, feat2) in permutations(self.sequence_features, 2):
            self.attention_layers[f"{feat1}_{feat2}"] = torch.nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.attn_norms[f"{feat1}_{feat2}"] = torch.nn.LayerNorm(d_model)
            cls_input_dim += d_model

        self.cls_mlp = torch.nn.Sequential(torch.nn.Linear(cls_input_dim, d_output))

        # Dropout layers for projections
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Encode to match feature dimensions
        x = {key: self.dropout(self.encoders[key](value)) for key, value in x.items()}

        if len(self.sequence_features) > 1:
            # Calculate Q, K, V
            queries = {modality: self.dropout(self.query_proj[modality](x[modality])) for modality in self.sequence_features}
            keys = {modality: self.dropout(self.key_proj[modality](x[modality])) for modality in self.sequence_features}
            values = {modality: self.dropout(self.value_proj[modality](x[modality])) for modality in self.sequence_features}
            
            # Apply multi-head cross-attention for each pair of modalities
            outputs = {}
            for (mod1, mod2) in permutations(self.sequence_features, 2):
                attn_output, _ = self.attention_layers[f"{mod1}_{mod2}"](queries[mod1], keys[mod2], values[mod2])
                attn_output = self.attn_norms[f"{mod1}_{mod2}"](attn_output)
                attn_output = self.dropout(attn_output)
                attn_output = attn_output.mean(1)
                outputs[f"{mod1}_{mod2}"] = attn_output

            # Get rid off encoded sequential features since they are processed by Multihead Attention
            x = {key: value.squeeze(1) for key, value in x.items() if key not in self.sequence_features}

            # Add the output features of Multihead Attention
            x.update(outputs)

        # Concatenate all
        x = torch.cat([x[key] for key in sorted(x.keys())], dim=1)
        x = self.relu(x)
        x = self.cls_mlp(x)
        
        return x