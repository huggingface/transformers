import os
import time

import cv2
import av
import numpy as np 
from numba import jit, cuda
from decord import VideoReader, cpu, gpu

import torch
from torchvision import io


video_dir = "/raid/raushan/temp_dir/"
NUM_FRAMES = 32


# @jit(nopython=True, target_backend='cuda') # <-- If you have a cuda GPU
def process_video_cv2(video: cv2.VideoCapture, indices: np.array, length: int):
    index = 0
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if index in indices:
            # Channel 0:B 1:G 2:R
            height, width, channel = frame.shape
            frames.append(frame[0:height, 0:width, 0:channel])
        if success:
            index += 1
        if index >= length:
            break

    video.release()
    return frames


def read_video_opencv(video_path, num_frames=NUM_FRAMES):
    '''
    Decode the video with open-cv decoder.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample uniformly. Defaults to NUM_FRAMES

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(int)
    frames = process_video_cv2(video, indices, total_num_frames)
    return np.stack(frames)



def read_video_decord(video_path, num_frames=NUM_FRAMES):
    '''
    Decode the video with Decord decoder.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample uniformly. Defaults to NUM_FRAMES

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    vr = VideoReader(uri=video_path, ctx=cpu(0)) # you need to install from source to use gpu ctx
    indices = np.arange(0, len(vr), len(vr) / num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def read_video_pyav(video_path, num_frames=NUM_FRAMES):
    '''
    Decode the video with PyAV decoder.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample uniformly. Defaults to NUM_FRAMES

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    container = av.open(video_path)

    # sample uniformly "num_frames" frames from the video
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)

    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])



def read_video_torchvision(video_path, num_frames=NUM_FRAMES):
    video, _, info = io.read_video(
        video_path,
        start_pts=0.0,
        end_pts=None,
        pts_unit="sec",
        output_format="TCHW",
    )

    idx = torch.linspace(0, video.size(0) - 1, num_frames, dtype=torch.int64)
    return video[idx]


decoders = {"decord": read_video_decord, "opencv": read_video_opencv, "av": read_video_pyav, "torchvision": read_video_torchvision}
for name, fn in decoders.items():
    start = time.perf_counter()
    for video_file in os.listdir(video_dir):
        path = f"{video_dir}/{video_file}"
        output = fn(path)

    end = time.perf_counter()
    print(f"Time taken for {name}: {(end-start):.04f} sec")


# Time taken for decord: 475.2979 sec
# Time taken for opencv: 614.6062 sec
# Time taken for av: 1067.0860 sec
# Time taken for torchvision: 1924.0433 sec

