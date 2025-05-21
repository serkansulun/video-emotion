import numpy as np
import torch
import yt_dlp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import ipywidgets as widgets
from IPython.display import display
import ffmpeg
import gdown
from scipy.stats import truncnorm
import requests
import os

def truncate_normal_dist(mean, std, low, high):
    return truncnorm(
        (low - mean) / std, (high - mean) / std, loc=mean, scale=std
    )


def wrap_text(text, linewidth):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 > linewidth:
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += " "
            current_line += word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def create_dropdown(options, description='Option:'):
    dropdown = widgets.Dropdown(options=options, description=description, description_width=700)
    display(dropdown)
    return dropdown


def draw_boxes_on_image(image, shapes, labels=None, output_file=None, title=None):
    # To visualize the face emotion and OCR results
    if labels == None:
        labels = [None] * len(shapes)
    if labels and len(labels) != len(shapes):
        raise ValueError("Number of labels must match the number of shapes.")

    fig, ax = plt.subplots()
    ax.imshow(image)
    
    for i, shape in enumerate(shapes):
        if len(shape) != 4:
            raise ValueError("Each shape must contain exactly 4 points.")
        
        box_color = 'yellow'
        # If a label is provided, add it below each shape with lime color and black outline
        if labels and labels[i] is not None:
            box_color = 'lime'
            centroid_x = np.mean([point[0] for point in shape])
            centroid_y = np.max([point[1] for point in shape]) + 10  # Place label below the shape
            
            # Draw the text with a black outline
            text = ax.text(centroid_x, centroid_y, labels[i], fontsize=6, color='lime',
                           ha='center', va='top', fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                                   path_effects.Normal()])
            
        # Create a polygon patch for each shape with a yellow outline and black stroke
        # shape = np.array(shape) + 0.2 * (np.array(shape) - np.mean(shape, axis=0))
        polygon = patches.Polygon(shape, closed=True, edgecolor=box_color, linewidth=1, fill=None)
        polygon.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                  path_effects.Normal()])
        ax.add_patch(polygon)
    
    ax.axis('off')
    if title != None:
        ax.set_title(title, fontdict={'fontsize': 8})
    if output_file != None:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.2)
    else:
        plt.show(block=False)


def convert_to_points(coords):
    # Converts points of corners to coordinates
    if len(coords) != 4:
        return []
    x1, y1, x2, y2 = coords
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def find_common_nonzero_argmax(boxes1, boxes2):
    # Get a sample frame where there are prefferably faces and text

    # Find number of boxes per frame
    n_ocr = np.array([len(frame_boxes) if frame_boxes != None else 0 for frame_boxes in boxes1], dtype=float)
    n_face = np.array([len(frame_boxes) if frame_boxes != [] else 0 for frame_boxes in boxes2], dtype=float)

    n_total = n_ocr + n_face

    # Replace 0 with minus infinity, so that we don't take those frames.
    n_ocr_inf = n_ocr.copy()
    n_ocr_inf[n_ocr==0] = -np.inf
    n_face_inf = n_face.copy()
    n_face_inf[n_face==0] = -np.inf

    n_total_inf = n_ocr_inf + n_face_inf

    # Find argmax (maximum number of boxes for face plus text)
    max_ind = np.argmax(n_total_inf)
    max_val = n_total_inf[max_ind]

    # If the maximum value is minus infinity, then get the maximum from original numbers.
    if max_val == -np.inf:
        max_ind = np.argmax(n_total)

    return max_ind


def get_video_duration(video_path):
    # Probe the video to get its duration
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='format=duration', format='json')
    except ffmpeg.Error as e:
        print(e.stderr.decode('utf-8'))
        raise
    # probe = ffmpeg.probe(video_path, select_streams='v:0', show_entries='format=duration', format='json')

    # probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='format=duration', format='json')
    duration = float(probe['format']['duration'])
    return duration


def extract_frames_and_audio(video_path, output_fps=None, size=None):
    # Probe the video to get original width, height, fps, and audio sampling rate
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    audio_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    
    original_width = int(video_info['width'])
    original_height = int(video_info['height'])
    original_fps = eval(video_info['r_frame_rate'])  # Get original FPS
    pix_fmt = video_info.get('pix_fmt', 'rgb24')

    if audio_info:
        audio_sample_rate = int(audio_info['sample_rate'])  # Get original audio sampling rate
        num_audio_channels = int(audio_info.get('channels', 1))
    else:
        audio_sample_rate = None
        num_audio_channels = None

    # --- GET FRAMES ---

    # Determine the number of channels based on the pixel format
    if 'gray' in pix_fmt:  # Black and white video
        num_channels = 1
        pix_fmt = 'gray'
    else:  # Color video
        num_channels = 3
        pix_fmt = 'rgb24'

    # Calculate new dimensions if resizing
    if size:
        if original_height < original_width:
            new_height = size
            new_width = int((original_width / original_height) * size)
        else:
            new_width = size
            new_height = int((original_height / original_width) * size)
    else:
        new_width = original_width
        new_height = original_height

    if output_fps == None:
        output_fps = original_fps
    
    out, _ = (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=output_fps)
        .filter('scale', new_width, new_height)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )

    # Calculate the number of frames
    num_frames = len(out) // (new_width * new_height * num_channels)

    # Convert the video frames to a numpy array
    video_array = np.frombuffer(out, np.uint8).reshape((num_frames, new_height, new_width, num_channels))

    # --- GET AUDIO ---

    if audio_info:      # if available
        bits = 32
        dtype_ = eval(f"np.int{bits}")
        max_ = np.iinfo(dtype_).max
        format = f"s{bits}le"

        out_audio, _ = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format=format)
            .run(capture_stdout=True)
        )
        # Convert the audio to a numpy array
        audio_array = np.frombuffer(out_audio, dtype=dtype_).astype(np.float32)
        audio_array = audio_array.reshape((-1, num_audio_channels)).T / max_
    else:
        audio_array = None

    return video_array, audio_array, original_fps, audio_sample_rate

def download_gdrive(id, target_path):
    target_path = str(target_path)
    url = f'https://drive.google.com/uc?id={id}&export=download'
    gdown.download(url, target_path, quiet=False)

def download_yt(url, target_dir, size=None):
    target_dir = str(target_dir)
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Set yt-dlp options
    ydl_opts = {
        'outtmpl': f'{target_dir}/youtube.%(ext)s',  # Save file in target path
        'noplaylist': True,  # Only download a single video
        'overwrites': True,  # Overwrite if the file exists
        'format': 'bestaudio/best',  # Default format selection
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
        'quiet': False,  # Show download progress
        'extract-audio': False,  # Do not extract audio (can be changed if needed)
    }
    
    # Apply size restriction if provided
    if size:
        ydl_opts['format'] = f'best[height<={size}]'  # Pick best format below given height

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
    except Exception as e:
        print("Error downloading video:", e)
        print("Falling back to best available format.")
        fallback_opts = ydl_opts.copy()
        del fallback_opts['format'] # Remove height restriction
        # fallback_opts['format'] = 'best'  # Try again with no height restriction
        with yt_dlp.YoutubeDL(fallback_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)

    # Get the file extension (default to mp4)
    ext = info_dict.get('ext', 'mp4')
    filename = f'{target_dir}/youtube.{ext}'
    
    return filename  # Return the saved file path

def download(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download file from {url}. Status code: {response.status_code}")


def equidistant_indices(source_length, target_length):
    assert source_length > target_length, 'Source length is less than target length.'
    inds = np.round(np.linspace(0, source_length - 1, num=target_length)).astype(np.int32).tolist()
    assert np.array_equal(inds, np.unique(inds))
    return inds


def normalize(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)

def detach_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        x.requires_grad = False
        x = x.cpu()
        x = x.numpy()
    return x