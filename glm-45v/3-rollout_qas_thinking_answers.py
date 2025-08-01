# python examples/test_glm4o/start_server_openai.py --config-path configs/100b-llm-moe --model-weight-path /workspace/wg/glm-train-prod/ckpt/100b-llm-moe-706/iter_0000079 --tensor-parallel-size 4 --enable-expert-parallel --t-patch-size 2 --max-num-seqs 1024 --max-image-num 0 --max-model-len 32768 --gpu-memory-utilization 0.91

# python examples/test_glm4o/start_server_openai.py --config-path configs/100b-vlm-moe-tp --model-weight-path /workspace/wg/glm-train-prod/ckpt/test_load  --original-model-weight-path /workspace/wg/glm-train-prod/ckpt/100b-vlm-exp-9-2/iter_0002000 --tensor-parallel-size 4 --enable-expert-parallel --t-patch-size 2 --max-num-seqs 1024 --max-image-num 300 --max-model-len 32768 --gpu-memory-utilization 0.91

import io
import os
import sys
import json
import time
import fcntl
import socket
import base64
import argparse
import traceback
import subprocess
import torch
import numpy as np
import math
from flask import Flask, request, jsonify
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from openai import OpenAI
import requests
import importlib.util
from decord import VideoReader

_path = os.path.dirname(os.path.dirname(__file__))
if _path not in sys.path:
    sys.path.append(_path)

from convert_100b_moe import load_specific_weights

import mmap
import struct
import collections

app = Flask(__name__)
client = None
model = None
inference_server = None

parser = argparse.ArgumentParser()

parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--model-weight-path', type=str, default=None)
parser.add_argument('--original-model-weight-path', type=str, default=None, help="original model weight path to be converted")
parser.add_argument('--data-parallel-size', type=int, default=1)
parser.add_argument('--data-parallel-size-local', type=int, default=1)
parser.add_argument('--tensor-parallel-size', type=int, default=1)
parser.add_argument('--enable-expert-parallel', action='store_true', default=True)
parser.add_argument('--pipeline-parallel-size', type=int, default=1)
parser.add_argument('--gpu-memory-utilization', type=float, default=0.9, help='Fraction of GPU memory to use')
parser.add_argument('--max-num-seqs', type=int, default=256, help='Maximum number of sequences to process in parallel')
parser.add_argument('--max-model-len', type=int, default=32768, help='Maximum sequence length the model can process')
parser.add_argument('--image-expect-length', type=int, default=6144)
parser.add_argument('--video-expect-length', type=int, default=6144)
parser.add_argument('--max-image-num', type=int, default=64, help='Maximum number of images to process')
parser.add_argument('--port', type=int, default=5002)
parser.add_argument('--t-patch-size', type=int, default=1, help='Size of temporal patch for video processing')

args, unknown_list = parser.parse_known_args()
args.inference_server_port = args.port - 1

"""
执行权重转换
"""

def acquire_conversion_lock(lock_file_path):
    """
    获取权重转换锁，确保只有一个进程执行转换
    """
    lock_fd = None
    try:
        # 创建锁文件
        lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
        
        # 尝试获取排他锁（非阻塞）
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        
        # 写入当前进程信息
        os.write(lock_fd, f"PID: {os.getpid()}, Time: {time.time()}".encode())
        os.fsync(lock_fd)
        
        print(f"Successfully acquired conversion lock: {lock_file_path}")
        return lock_fd
    except (OSError, IOError) as e:
        # 锁已被其他进程持有
        if lock_fd:
            os.close(lock_fd)
        return None

def release_conversion_lock(lock_fd, lock_file_path):
    """
    释放权重转换锁
    """
    try:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        
        # 删除锁文件
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)
        
        print(f"Released conversion lock: {lock_file_path}")
    except Exception as e:
        print(f"Warning: Failed to release lock {lock_file_path}: {e}")

def wait_for_conversion_completion(complete_path, check_interval=10):
    """
    等待权重转换完成
    """
    start_time = time.time()
    
    while True:
        # 检查是否有完成标志
        if os.path.exists(complete_path):
            return True
        print(f"Waiting for model weight conversion to complete... ({int(time.time() - start_time)}s elapsed)")
        time.sleep(check_interval)

if args.original_model_weight_path:
    complete_path = args.model_weight_path + "/.convert_complete"
    if os.path.exists(args.model_weight_path) and os.path.exists(complete_path):
        print(f"Model weight path {args.model_weight_path} already exists, skip conversion")
    else:
        os.makedirs(args.model_weight_path, exist_ok=True)
        # 创建锁文件路径
        lock_file_path = f"{args.model_weight_path}/.convert_lock"
        
        # 尝试获取转换锁
        lock_fd = acquire_conversion_lock(lock_file_path)
        
        if lock_fd is not None:
            # 当前进程获得锁，执行转换
            try:
                print(f"Converting model weight from {args.original_model_weight_path} to {args.model_weight_path}")
                dicts_to_save = load_specific_weights(
                    checkpoint_dir=args.original_model_weight_path, 
                    output_dir=args.model_weight_path, 
                    target_ep=4
                )
                with open(complete_path, "w") as f:
                    f.write("convert_complete")
                print(f"Model weight conversion completed successfully")
            except Exception as e:
                print(f"Error during model weight conversion: {e}")
                raise
            finally:
                # 无论成功还是失败，都要释放锁
                release_conversion_lock(lock_fd, lock_file_path)
        else:
            # 其他进程正在转换，等待完成
            print(f"Another process is converting model weights, waiting...")
            success = wait_for_conversion_completion(complete_path)
            if not success:
                raise RuntimeError(f"Failed to wait for model weight conversion completion")
"""
开始服务逻辑
"""

TarHeader = collections.namedtuple(
    "TarHeader",
    [
        "name",
        "mode",
        "uid",
        "gid",
        "size",
        "mtime",
        "chksum",
        "typeflag",
        "linkname",
        "magic",
        "version",
        "uname",
        "gname",
        "devmajor",
        "devminor",
        "prefix",
    ],
)


def parse_tar_header(header_bytes):
    """解析 tar 格式的文件头信息
    Args:
        header_bytes (bytes): header bytes, less than 500
    Returns:
        tar header info
    """
    assert len(header_bytes) == 500, f"tar header length must be 500, but found {header_bytes}"
    header = struct.unpack("!100s8s8s8s12s12s8s1s100s6s2s32s32s8s8s155s", header_bytes)
    return TarHeader(*header)


def extract_data_from_tarfile(tar_path, offset):
    """根据偏移量从tar流中获取数据
    Args:
        tar_path (str): tar path
        offset (int): offset
    Returns:
        name, 
        data bytes
    """
    try:
        with open(tar_path, "rb") as stream:
            stream = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            header = parse_tar_header(stream[offset: offset + 500])
            name = header.name.decode("utf-8").strip("\x00")
            start = offset + 512
            end = start + int(header.size.decode("utf-8")[:-1], 8)
            return name, stream[start: end]
    except:
        print(f"Failed: {tar_path}, offset: {offset}")
        print(traceback.format_exc())

def close_inference_server():
    if inference_server:
        # 终止进程
        inference_server.terminate()
        inference_server.wait()
        # 清空显存占用
        os.system("fuser -k /dev/nvidia*")
    print("Inference server has been killed!")

def connect_inference_server(port, host="127.0.0.1") -> bool:
    def connected():
        try:
            # 创建一个TCP套接字
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(10)  # 设置超时时间，防止长时间等待
                result = sock.connect_ex((host, port))  # 尝试连接到指定的主机和端口
                if result == 0:
                    return True
                else:
                    return False
        except socket.error as e:
            return False

    _seconds = 10
    while not connected():
        time.sleep(_seconds)
        print(f"Not connected, will try again after {_seconds}s")
        
    print(f"Connected!")
    global client, model
    client = OpenAI(
        api_key="sk-",
        base_url=f"http://{host}:{port}/v1",
    )
    
    model = client.models.list().data[0].id

def start_inference_server():
    assert os.path.exists(args.model_weight_path), f"Not found {args.model_weight_path}!"
    # print("Close old server...")
    # close_inference_server()

    vllm_server_cmd = [
        "vllm", "serve",
        args.config_path,
        "--model-weight-path", args.model_weight_path,
        "--tensor-parallel-size", args.tensor_parallel_size,
        "--pipeline-parallel-size", args.pipeline_parallel_size,
        "--data-parallel-size", args.data_parallel_size,
        "--data-parallel-size-local", args.data_parallel_size_local,
        "--port", args.inference_server_port,
        "--load-format", "megatron",
        "--limit-mm-per-prompt", f"image={args.max_image_num},video=0",
        "--max-model-len", args.max_model_len,
        "--trust-remote-code",
        "--max-num-seqs", args.max_num_seqs,
        "--gpu-memory-utilization", args.gpu_memory_utilization,
        "--allowed-local-media-path", '/',
        "--compilation-config", '{"level": 3, "use_cudagraph": true}',
        "--disable-mm-preprocessor-cache"
    ] + unknown_list
    
    if args.enable_expert_parallel:
        vllm_server_cmd.append("--enable-expert-parallel")

    vllm_server_cmd = [str(v) for v in vllm_server_cmd]

    print("Start new server...")
    print(f"Run: {' '.join(vllm_server_cmd)}")

    global inference_server
    mode = "w" if inference_server is None else "a"
    log_file = "inference_server.log"
    with open(log_file, mode) as fp:
        inference_server = subprocess.Popen(vllm_server_cmd, stdout=fp, stderr=fp)
        print(f"Log saved to {log_file}. If the server is not started for a long time, please check the log file.")
    connect_inference_server(args.inference_server_port, host="127.0.0.1")


def post_query(**kwargs):
    data = {
        **kwargs
    }
    headers = {'Content-Type': 'application/json'}

    response = requests.post(
        f"http://localhost:5001/v1/chat/completions",
        headers=headers,
        data=json.dumps(data),
        verify=False
    )
    
    return response

def predict(request_data):
    force_thinking_token_nums = request_data.pop('force_thinking_token_nums', 0)
    n = request_data.get("n", 1)
    raw_response = post_query(**request_data)
    try:
        response = raw_response.json()
        
        # Support n parameters
        if n > 1:
            answers = []
            for choice in response["choices"]:
                answer = choice["message"]["content"]
                answers.append(answer)
            return answers, 200
        else:
            answer = response["choices"][0]["message"]["content"]
            if force_thinking_token_nums and "</think>" not in answer:
                print(f"force_thinking_token_nums: {force_thinking_token_nums}, answer: {answer}")
                
                answer = answer + "</think>"
                    
                # 续写
                request_data['messages'].append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer
                        }
                    ]
                })
                
                request_data["max_tokens"] = force_thinking_token_nums
                response = post_query(**request_data)
                new_answer = response.json()["choices"][0]["message"]["content"]
                answer = answer + new_answer
            return answer, 200
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        return str(e), raw_response.status_code

def smart_resize(
        t: int,  h: int, w: int, t_factor: int = 1, h_factor: int = 28, w_factor: int = 28, 
        min_pixels: int = 112 * 112, max_pixels: int = 14 * 14 * 4 * 15000, 
    ):
    """
    copy from qwen2vl
    https://github.com/huggingface/transformers/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
    Rescales the image so that the following conditions are met:

    1. Both dimensions (h and w) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    
    assert t >= t_factor, \
        'Temporal dimension must be greater than the factor.'
        
    h_bar = round(h / h_factor) * h_factor
    w_bar = round(w / w_factor) * w_factor
    t_bar = round(t / t_factor) * t_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((t * h * w) / max_pixels)
        h_bar = math.floor(h / beta / h_factor) * h_factor
        w_bar = math.floor(w / beta / w_factor) * w_factor
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = math.ceil(h * beta / h_factor) * h_factor
        w_bar = math.ceil(w * beta / w_factor) * w_factor
        
    return h_bar, w_bar

def load_image_to_base64_image(image_path, t_patch_size: int, max_pixels: int):
    if isinstance(image_path, bytes):
        image_data = image_path
    elif image_path.startswith("data:image/"):
        image_data = base64.b64decode(image_path.split(",")[1])
    elif os.path.isfile(image_path):
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
    else:
        raise ValueError("Invalid image path or data.")

    image = Image.open(io.BytesIO(image_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # 原始图像的高度和宽度
    w, h = image.size

    # 计算新的尺寸
    h_bar, w_bar = smart_resize(
        t=t_patch_size, 
        h=h, w=w, 
        t_factor=t_patch_size,
        h_factor=14 * 2,
        w_factor=14 * 2,
        min_pixels=112 * 112,
        max_pixels=max_pixels
    )
    # 调整图像大小
    image = image.resize((w_bar, h_bar), Image.Resampling.BICUBIC)

    # 将调整大小后的图像保存为字节数据
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    image_data = buffered.getvalue()
    ####################################
    
    # 将图像数据转换为Base64编码
    base64_encoded_data = base64.b64encode(image_data)
    image_base64 = base64_encoded_data.decode('utf-8')

    return image_base64

def process_image_file(image_path: str, t_patch_size: int, max_pixels: int):
    if os.path.isfile(image_path):
        encoded_image = load_image_to_base64_image(image_path, t_patch_size=t_patch_size, max_pixels=max_pixels)
    elif image_path.startswith("data:image/"):
        encoded_image = load_image_to_base64_image(image_path, t_patch_size=t_patch_size, max_pixels=max_pixels)
    elif image_path.startswith("<|base64|>"):
        img_bytes = base64.b64decode(image_path[len("<|base64|>"):])
        encoded_image = load_image_to_base64_image(img_bytes, t_patch_size=t_patch_size, max_pixels=max_pixels)
    elif image_path.startswith("<|tarpath|>"):
        image_path = image_path[len("<|tarpath|>"):]
        tar_path, tar_offset = image_path.split("<|offset|>")
        _, img_data = extract_data_from_tarfile(tar_path, int(tar_offset))
        encoded_image = load_image_to_base64_image(img_data, t_patch_size=t_patch_size, max_pixels=max_pixels)
    else:
        raise FileNotFoundError(f"{image_path} is not a valid file path!")

    return encoded_image


def read_frames(
    video_path, 
    num_frames, 
    frame_indexes, 
    t_patch_size: int,
    max_pixels: int,
    fps=2, 
    frame_method='mrope'
):
    assert num_frames is None or frame_indexes is None, "num_frames and frame_indexes cannot be set at the same time."
    inv_fps = 1 / fps
    if isinstance(video_path, str):
        video_reader = VideoReader(video_path)
    elif isinstance(video_path, bytes):
        video_reader = VideoReader(io.BytesIO(video_path))
    else:
        raise ValueError(f"video_path must be str or bytes, got {type(video_path)}")

    vlen = len(video_reader)
    print("vlen:", vlen)
    video_fps = video_reader.get_avg_fps()
    total_frames = vlen
    timestamps = video_reader.get_frame_timestamp(np.arange(total_frames))
    timestamps = [i[0] for i in timestamps]
    duration = round(max(timestamps)) + 1
    second_idxs = None
    if num_frames is None:
        frame_indices = frame_indexes
        if max(frame_indices) >= total_frames:
            raise ValueError(f"frame_indexes {frame_indexes} exceed the total frames {total_frames}.")
        print("frame_indices:", frame_indices)
        if len(frame_indexes) == 1 and frame_indexes[0] < 0:
            # implies new avg sampling strategy, sample averagely and the second index is float with one decimal
            # copied from glm/data/datasets/utils.py read_frames_decord_avg
            avg_num_frames = -frame_indexes[0]
            second_indices = np.linspace(0, duration, avg_num_frames, endpoint=True)
            frame_indices = []
            current_second_index = 0
            for i in range(vlen):
                if timestamps[i] >= second_indices[current_second_index]:
                    frame_indices.append(i)
                    current_second_index += 1
                    if current_second_index == len(second_indices):
                        break
            if len(frame_indices) < avg_num_frames:
                print(
                    f"Warning: video {video_path} has only {len(timestamps)} frames, but {avg_num_frames} frames are required. Padding with last frame.")
                last_frame_idx = frame_indices[-1]
                frame_indices += [last_frame_idx] * (avg_num_frames - len(frame_indices))
            # directly change sceond_indices to str, copied from create_ffr_avg.py
            second_idxs = ["{:.1f}".format(second) for second in second_indices]
    else:
        print("num_frames:", num_frames, "duration:", duration)
        if num_frames >= duration * fps:
            # extract at 1 fps
            frame_indices = []
            current_second = 0
            for frame_index in range(total_frames):
                if timestamps[frame_index] >= current_second:
                    current_second += inv_fps
                    frame_indices.append(frame_index)
                    if current_second >= duration:
                        break
        else:
            # uniformly extract num_frames frames
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)


    if frame_method == 'mrope':
        # the frame_indices and the second_idxs must be even
        if len(frame_indices) % 2 != 0:
            # padd with the last frame
            frame_indices.append(frame_indices[-1])
            
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), uint8
    
    ####################################
    # SMART RESIZE
    T, H, W, _ = frames.shape
    # 调整维度从 (T, H, W, C) 到 (T, C, H, W)
    frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2)  # (T, C, H, W), uint8
    
    # 使用smart_resize计算新的大小
    new_h, new_w = smart_resize(
        T, H, W, 
        t_factor=t_patch_size,
        h_factor=14 * 2,
        w_factor=14 * 2,
        min_pixels=112 * 112,
        max_pixels=max_pixels
    ) # fix here
    frames = F.interpolate(frames, size=(new_h, new_w), mode='bicubic', align_corners=False)
    ####################################
    
    if second_idxs is None:
        second_idxs = [int(timestamps[f]) for f in frame_indices]
    print("frame_indices:", frame_indices)
    print("second_idxs:", second_idxs)
    
    return frames, second_idxs, float(video_fps)

def load_video_to_base64_images(
    video_path, 
    t_patch_size: int,
    max_pixels: int,
    num_frames=300, 
    frame_indexes=None, 
    fps=2, 
    frame_method='mrope'
):
    assert num_frames is None or frame_indexes is None, "num_frames and frame_indexes cannot be set at the same time."
    frames, second_idxs, _ = read_frames(video_path, num_frames, frame_indexes, fps=fps, frame_method=frame_method, t_patch_size=t_patch_size, max_pixels=max_pixels)
    # print(second_idxs)
    if frame_method == 'mrope':
        assert fps == 2, "fps must be 2 when using mrope method."
        out_second_idxs = []
        for i in range(0, len(second_idxs), 2):
            out_second_idxs.append(second_idxs[i])
        second_idxs = out_second_idxs   
    base64_images = [image_tensor_to_base64(frames[i]) for i in range(frames.shape[0])]
    return base64_images, second_idxs

def process_video_file(
    video_path: str, 
    max_pixels: int,
    t_patch_size: int,
    num_frames: int = 300,
    frame_indexes: list[int] = None,
    fps: int = 2,
    frame_method: str = 'mrope'
):
    if os.path.isfile(video_path):
        video_images, second_idxs = load_video_to_base64_images(
            video_path,
            t_patch_size=t_patch_size,
            num_frames=num_frames,
            frame_indexes=frame_indexes,
            fps=fps, 
            frame_method=frame_method, 
            max_pixels=max_pixels)
    elif video_path.startswith("data:video/"):
        # 处理data:video格式的base64编码
        video_data = base64.b64decode(video_path.split(",")[1])
        video_images, second_idxs = load_video_to_base64_images(
            video_data,
            t_patch_size=t_patch_size,
            num_frames=num_frames,
            frame_indexes=frame_indexes,
            fps=fps, 
            frame_method=frame_method, 
            max_pixels=max_pixels)
    elif video_path.startswith("<|base64|>"):
        # 处理<|base64|>前缀的base64编码
        video_bytes = base64.b64decode(video_path[len("<|base64|>"):])
        video_images, second_idxs = load_video_to_base64_images(
            video_bytes,
            t_patch_size=t_patch_size,
            num_frames=num_frames,
            frame_indexes=frame_indexes,
            fps=fps, 
            frame_method=frame_method, 
            max_pixels=max_pixels)
    elif video_path.startswith("<|tarpath|>"):
        # 处理tar文件中的视频
        video_path = video_path[len("<|tarpath|>"):]
        tar_path, tar_offset = video_path.split("<|offset|>")
        _, video_data = extract_data_from_tarfile(tar_path, int(tar_offset))
        video_images, second_idxs = load_video_to_base64_images(
            video_data,
            t_patch_size=t_patch_size,
            num_frames=num_frames,
            frame_indexes=frame_indexes,
            fps=fps, 
            frame_method=frame_method, 
            max_pixels=max_pixels)
    else:
        raise FileNotFoundError(f"{video_path} is not a valid file path!")

    return video_images, second_idxs


def image_tensor_to_base64(image_tensor):
    if image_tensor.shape[0] != 3:
        raise ValueError("Input tensor is not a 3-channel image.")
    image_array = image_tensor.permute(1, 2, 0).numpy()
    image_array = image_array.astype(np.uint8)
    image = Image.fromarray(image_array)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def process_single_msg(
    msg: dict[str, str | list], 
    image_expect_length: int, 
    video_expect_length: int, 
    temporal_patch_size : int =1,
    frame_method: str = 'mrope'
) -> dict[str, str | list]:
    msg_content: list[dict] = msg["content"]

    processed_single_msg = []
    text_input = ""
    for content in msg_content:
        if content["type"] == "text":
            text_placeholder = content["text"]
            
        elif content["type"] == "image_url":
            text_placeholder = "<|begin_of_image|><|image|><|end_of_image|>"
            image_url = content["image_url"]["url"]
            encoded_image = process_image_file(image_url, t_patch_size=temporal_patch_size, max_pixels=image_expect_length * 14 * 14 * 2 * 2 * temporal_patch_size)
            new_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            }
            
            for _ in range(temporal_patch_size):
                processed_single_msg.append(new_content)   
                
        elif content["type"] == "video_url":
            video_url = content["video_url"]["url"]
            video_images, second_idxs = process_video_file(
                video_url, 
                t_patch_size=temporal_patch_size, 
                max_pixels=video_expect_length * 14 * 14 * 2 * 2 * temporal_patch_size,
                frame_method=frame_method
            )
            multi_modal_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                    },
                } for image_base64 in video_images
            ]
            processed_single_msg.extend(multi_modal_content)

            # text placeholder
            text_placeholder = "<|begin_of_video|>"
            for second in second_idxs:
                text_placeholder += "<|begin_of_image|><|image|><|end_of_image|>" + str(second)
            text_placeholder += "<|end_of_video|>"
            
        else:
            raise ValueError(f"{content['type']} is not a valid type.")

        text_input += text_placeholder
        
    print("text_input:", text_input)  
    processed_single_msg.append({
        "type": "text",
        "text": text_input
    })

    return {
        "role": "user",
        "content": processed_single_msg
    }
    

@app.route('/v1/chat/completions', methods=['POST'])
def generate():
    # 验证 Content-Type 是否为 application/json
    if request.headers.get('Content-Type') != 'application/json':
        return jsonify({"error": "Invalid Content-Type. Expected 'application/json'."}), 400

    # 获取 JSON 数据
    try:
        data = request.json
    except Exception as e:
        return jsonify({"error": "Invalid JSON payload"}), 400
    
    messages = data['messages']
    
    try:
        processed_messages = []
        for msg in messages:
            if msg["role"] == "system":
                processed_messages.append(msg)
            elif msg["role"] == "user":
                # change the temporal_patch_size if ckpt is t-patch
                single_msg = process_single_msg(
                    msg, 
                    temporal_patch_size=args.t_patch_size,
                    image_expect_length=args.image_expect_length,
                    video_expect_length=args.video_expect_length,
                    frame_method='mrope'
                )
                processed_messages.append(single_msg)
            elif msg["role"] == "assistant":
                processed_messages.append(msg)
            else:
                raise ValueError(f"{msg['role']} is not a valid role for a message.")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"处理message信息时出错：\n{e}"}), 500

    data['messages'] = processed_messages
    
    try:
        answer, status_code = predict(data)
        
        if status_code == 200:
            if isinstance(answer, str):
                return jsonify({"choices": [{"message": {"content": answer.strip()}}]}), 200
            elif isinstance(answer, list):
                return jsonify({"choices": [{"message": {"content": (a.strip() if isinstance(a, str) else str(a))}} for a in answer]}), 200
        else:
            return answer, status_code
    except:
        traceback.print_exc()
        return jsonify({"error": "处理图片时出现错误"}), 500


@app.route('/new_inference_server', methods=['POST'])
def new_inference_server():
    if 'model_weight_path' not in request.form:
        print("Missing model_weight_path.")
        return "Missing model_weight_path", 500
    model_weight_path = request.form["model_weight_path"]

    global args
    if model_weight_path == args.model_weight_path:
        print("Server has started!")
        return "skip"

    args.model_weight_path = model_weight_path
    start_inference_server()

    print(f"Checkout to new inference server: {args.model_weight_path}!")

    return "update"


@app.route('/shutdown_inference_server', methods=['POST'])
def shutdown_inference_server():
    if args.model_weight_path is not None:
        print("Close old server...")
        close_inference_server()
        return "shutdown"
    else:
        return "skip"


@app.route('/get_model_iterid')
def get_model_iterid():
    if args.model_weight_path is None:
        return ""
    else:
        return args.model_weight_path


if __name__ == '__main__':
    if args.model_weight_path is not None:
        start_inference_server()
    else:
        print(
            f'Not specified --model-weight-path, will start a empty flask server.\nPlease start a inference server using """curl -X POST -d "model_weight_path=/workspace/ckpt/wy/checkpoints/glm-train-GLM32B-sft1-1344-0826-1900/iter_0000001"http://localhost:5002/new_inference_server""" before using inference api.')
    app.run(debug=False, host="0.0.0.0", port=args.port)