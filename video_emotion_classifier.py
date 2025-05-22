import sys
import os
from pathlib import Path

import zipfile
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
import gdown
import argparse
from time import time
t0 = time()
print('Importing libraries...', flush=True, end=' ')
sys.path.append('src')
import utils as u
import models_
print(f'Done.', flush=True)

# print('Done.', flush=True)

parser = argparse.ArgumentParser(description="Process a video for emotion classification.")
parser.add_argument('--video_path', type=str, default=None, help="Path to the input video file.")
parser.add_argument('--youtube_link', type=str, default=None, help="YouTube link of the video.")
parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
args = parser.parse_args()

video_path = args.video_path
youtube_link = args.youtube_link

size = 360   # Use 360p for YouTube downloads

if video_path is None and youtube_link is None:
    video_path = 'sample_data/sample.mp4'
    print(f"No file or link provided. Defaulting to {video_path}")
elif video_path is not None and youtube_link is not None:
    print("Both file and link provided. Using the file.")
elif youtube_link is not None:
    print('Downloading video from YouTube...', flush=True, end=' ')
    video_path = u.download_yt(youtube_link, target_dir='sample_data', size=size)
    print('Done.', flush=True)


# # Check if weights folder exists, if not, create it
# weights_folder = Path('weights')
# if not weights_folder.exists():
#     # Download weights
#     weights_url = 'https://drive.google.com/uc?id=1uvFdB55W7vTYlfsTsK4VhZkB4vCYStf_&export=download'
#     print('Downloading weights...', flush=True, end=' ')
#     gdown.download(weights_url, 'weights.zip', quiet=False)

#     # Unzip the downloaded file
#     with zipfile.ZipFile('weights.zip', 'r') as zip_ref:
#         zip_ref.extractall('.')

#     # Delete the zip file
#     os.remove('weights.zip')
#     print('Done.', flush=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

visualize = True    # Visualize intermediate results

# save_memory = False

labels = ('anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise')

model_dir = Path('weights/classifier')
config = torch.load(model_dir / 'model_config.pt', weights_only=False)

print('Loading model...', flush=True)
# Load model
classifier = models_.AttentionClassifier(
        feature_dims=config['feature_dims'],
        # n_layers=config['n_layers'],
        d_model=config['d_model'],
        d_output=config['n_labels'],
        dropout=config['dropout'],
        )

weights = torch.load(model_dir / 'model.pt', weights_only=True, map_location=lambda storage, loc: storage)
classifier.load_state_dict(weights)
del weights
print(f"Model loaded from {model_dir}/model.pt", flush=True)
classifier = classifier.to(device)
classifier.eval()

feature_names = config['features']

n_frames = config['n_frames']

stats = torch.load('stats.pt', weights_only=False)

extractors = {}
# extractors['face_emotion'] = models_.FaceExtractAndClassify()

print('Loading feature extractors:', end=' ', flush=True)
for feature_name in feature_names:
    print(feature_name, end=', ', flush=True)
    if feature_name == 'clip':
        extractors[feature_name] = models_.CLIPRunner()
    elif feature_name == 'beats':
        extractors[feature_name] = models_.BEATSRunner()
    elif feature_name == 'asr_sentiment':
        extractors[feature_name] = models_.ASRSentiment()
    elif feature_name == 'ocr_sentiment':
        extractors[feature_name] = models_.OCRPipeline()
    elif feature_name == 'face_emotion':
        extractors[feature_name] = models_.FaceExtractAndClassify()
    extractors[feature_name].to_device(device)
print('caption', end=', ', flush=True)
caption_model = models_.CaptionRunner()

print('Done.', flush=True)

# Load video

print('Extracting video frames and audio...', end=' ', flush=True)

# The visual models use a fixed number of frames (16).
# So adjust FPS to extract only that many frames.
video_duration = u.get_video_duration(video_path)
input_frame_sequence_length = config['feature_lengths']['clip']
output_fps = input_frame_sequence_length / video_duration

input_frames, input_audio, input_fps, sr = u.extract_frames_and_audio(str(video_path), output_fps=output_fps, size=size)

print('Done.', flush=True)

sample_features = {}
video_outputs = {}
clip_features = None    # We will use it later for captioning

# PART II - FEATURE EXTRACTION
with torch.no_grad():
    for feature_name in tqdm(feature_names, desc='Extracting features'):

        print(f'\nFeature: {feature_name}', flush=True)
        if feature_name == 'clip':
            input_tensor = input_frames
        elif feature_name == 'beats':
            input_tensor = input_audio
        elif feature_name == 'asr_sentiment':
            input_tensor = input_audio
        elif feature_name == 'ocr_sentiment':
            input_tensor = input_frames
        elif feature_name == 'face_emotion':
            input_tensor = input_frames

        video_output = extractors[feature_name].process_video(input_tensor=input_tensor, n_frames=n_frames, sr=sr)
        extracted_feature = video_output['features']

        if feature_name == 'clip':
            clip_features = extracted_feature

        # Keep other output for visualization
        video_outputs[feature_name] = video_output

        if extracted_feature == []:     # Feature unavailable, use zeros
            extracted_feature = torch.zeros((config['feature_lengths'][feature_name],
                config['feature_dims'][feature_name]))
        elif config['normalize_data']:      # Normalization
            extracted_feature = u.normalize(extracted_feature, stats[feature_name]['min'], stats[feature_name]['max'])

        # Fix lengths
        source_length = extracted_feature.shape[0]
        target_length = config['feature_lengths'][feature_name]

        if source_length > target_length:     # Take equidistant frames
            inds = u.equidistant_indices(source_length, target_length)
            extracted_feature = extracted_feature[inds, :]
        elif source_length < target_length:
            extracted_feature = torch.nn.functional.pad(extracted_feature, (0, 0, 0, target_length - source_length))

        # Add batch dimension and move to device
        extracted_feature = extracted_feature.unsqueeze(0).to(device)

        sample_features[feature_name] = extracted_feature

    # PART III - RUN MODEL
    with torch.cuda.amp.autocast(enabled=config['amp']):
        output = classifier(sample_features).squeeze()
        del sample_features
        output = torch.nn.functional.softmax(output)
        output = u.detach_tensor(output)


# PART IV - VISUALIZATION (OPTIONAL)
if visualize:

    # *** FACE and OCR
    # Display a sample frame with OCR and facial boxes (later)
    ocr_boxes = video_outputs['ocr_sentiment']['boxes']
    face_boxes = video_outputs['face_emotion']['coordinates']

    # Find the frame with faces and text
    i = u.find_common_nonzero_argmax(ocr_boxes, face_boxes)

    if ocr_boxes == None or ocr_boxes[i] == None:
        ocr_boxes = []
    else:
        # Get OCR boxes for that frame
        ocr_boxes = [pred[0] for pred in ocr_boxes[i]]

    # We don't need predictions per frame, we will print OCR from the entire video
    ocr_predictions = [None] * len(ocr_boxes)

    if face_boxes[i] == None:
        face_boxes[i] = []

    face_boxes = [u.convert_to_points(box) for box in face_boxes[i]]
    face_predictions = video_outputs['face_emotion']['predictions'][i]
    if face_predictions == None:
        face_predictions = []

    face_predictions = [f'{round(pred[1] * 100):2}%  {pred[0].upper()}' for pred in face_predictions]

    boxes = face_boxes + ocr_boxes
    box_labels = face_predictions + ocr_predictions

    selected_frame = input_frames[i]

    # *** CLIP caption
    # Using the same sample frame, create a caption using CLIP
    clip_feature = clip_features[i:i+1, ...].to(device)
    with torch.no_grad():
        caption_model.to_device(device)   # Move to GPU if exists
        caption = caption_model(clip_feature).capitalize()

    # *** Optical Character Recognition (OCR) analysis
    ocr_text = video_outputs['ocr_sentiment']['ocr_processed']
    if ocr_text != []:
        ocr_text = '. '.join(video_outputs['ocr_sentiment']['ocr_processed'])
        sentiment_prediction = video_outputs['ocr_sentiment']['predictions'][0].capitalize()
        sentiment_percentage = round(video_outputs['ocr_sentiment']['predictions'][1] * 100)
        print()
        print('Optical character recognition (OCR):', flush=True)
        print(u.wrap_text(ocr_text, 72), flush=True)
        print('Sentiment analysis:', flush=True)
        print(f"{sentiment_percentage}%  {sentiment_prediction}", flush=True)

    # *** Automatic Speech Recognition (ASR) analysis
    asr_text = video_outputs['asr_sentiment']['asr']
    asr_language = video_outputs['asr_sentiment']['language']
    if asr_text != "":
        sentiment_prediction = video_outputs['asr_sentiment']['predictions'][0].capitalize()
        sentiment_percentage = round(video_outputs['asr_sentiment']['predictions'][1] * 100)
        print()
        print('Automatic speech recognition (ASR):')
        if asr_language not in ('english', None):
            print(f'(Translated from {asr_language.capitalize()}.), flush=True')
        print(u.wrap_text(asr_text, 72), flush=True)
        print('Sentiment analysis:', flush=True)
        print(f"{sentiment_percentage}%  {sentiment_prediction}", flush=True)

    # *** BEATS classification
    # Using the entire video, print BEATS (audio event) probabilities
    predictions = video_outputs['beats']['predictions']
    if predictions != []:
        print()
        print('Audio classification:', flush=True)
        best_prediction = predictions[0]
        print(f"{round(best_prediction[1] * 100)}%  {best_prediction[0]}")
        for prediction in predictions[1:]:
            print(f'{round(prediction[1] * 100)}%  {prediction[0]}')
        print()

    # *** FACE and OCR
    # Display a sample frame with OCR and facial boxes
    no_display = matplotlib.get_backend() == 'agg'
    title = 'Predicted caption: ' + caption
    if no_display:
        output_file = 'sample_frame.png'
    else:
        output_file = None
        print('Sample frame', flush=True)
    u.draw_boxes_on_image(selected_frame, boxes, labels=box_labels, output_file=output_file, title=title)
    if no_display:
        print('Sample frame saved to', output_file, flush=True)


# Print emotion prediction results
sorted_indices = np.argsort(output)[::-1]

print()
print('Emotion prediction: ', flush=True)
for idx in sorted_indices:
    print(f"{round(output[idx] * 100)}%  {labels[idx].capitalize()}", flush=True)