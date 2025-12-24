# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import base64
import copy
import re
import io
import json
import os
import os.path as osp
import random
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Sequence
import math
import numpy as np
import PIL
import torch
import transformers
from PIL import Image, ImageFile
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer
from transformers import AutoFeatureExtractor
import kaldiio
import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import DEFAULT_SOUND_TOKEN,DEFAULT_SPEECH_TOKEN, IGNORE_INDEX
from llava.data.collate import DataCollator
from llava.mm_utils import (
    load_audio,
    get_num_windows,
    tokenizer_image_token,
)
from llava.train.args import DataArguments, TrainingArguments
from llava.train.sequence_parallel import (
    extract_local_from_list,
    extract_local_input_ids,
    extract_local_position_ids,
    get_pg_manager,
)
from llava.utils.tokenizer import preprocess_conversation
# import torchaudio
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler, UniformClipSampler
import soundfile as sf
from librosa import resample as librosa_resample
import whisper
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000
import soundfile as sf
from librosa import resample as librosa_resample
import whisper
import librosa
from llava.utils.logging import logger
import os
from pydub import AudioSegment
import subprocess
import random
import io
import numpy as np
import librosa
from decord import AudioReader, cpu
import datetime
# SwiftStack / OpenStack credentials and endpoints
os.environ["ST_USER"] = "arushig"
os.environ["ST_KEY"] = ""
os.environ["ST_END_POINT"] = "https://pdx.s8k.io"
os.environ["ST_AUTH"] = "https://pdx.s8k.io/auth/v1.0"
os.environ["OS_STORAGE_URL"] = "https://pdx.s8k.io/v1/AUTH_team-nemo-data-acquisition"
# import torch
# torch.distributed.init_process_group(
#     backend="nccl",
#     timeout=datetime.timedelta(minutes=30)
# )
from llava.data.audio_utils import _load_tarball_local_audio_byteseek
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

import warnings

MAX_DURATION = 1000  # seconds
warnings.filterwarnings("ignore", category=FutureWarning)

def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        concat_values = "".join([sentence["value"] for sentence in source])
        for sid, sentence in enumerate(source):
            # In multimodal conversations, we automatically prepend '<image>' at the start of the first sentence if it doesn't already contain one.
            
            if DEFAULT_SOUND_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_SOUND_TOKEN, f"{DEFAULT_SOUND_TOKEN}\n")
                sentence["value"] = sentence["value"].replace(f"{DEFAULT_SOUND_TOKEN}\n\n", f"{DEFAULT_SOUND_TOKEN}\n")
            if DEFAULT_SPEECH_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_SPEECH_TOKEN, f"{DEFAULT_SPEECH_TOKEN}\n")
                sentence["value"] = sentence["value"].replace(f"{DEFAULT_SPEECH_TOKEN}\n\n", f"{DEFAULT_SPEECH_TOKEN}\n")
    return sources


def read_audio_from_ss(container, object_name):
    
    

    def read_swift_file(container, object_name):
        try:
            result = subprocess.run(
                ["swift", "download", container, object_name, "--output", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return result.stdout  # returns bytes
        except subprocess.CalledProcessError as e:
            print("Error reading Swift object:", e.stderr.decode())
            return None

   

    data = read_swift_file(container, object_name)

    return data

def get_mp4_duration_bytes(mp4_bytes):
    """
    Get duration (in seconds) of MP4 audio bytes using ffprobe.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            "pipe:0"
        ]
        proc = subprocess.run(
            cmd,
            input=mp4_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        info = json.loads(proc.stdout)
        duration = float(info["format"]["duration"])
        return duration
    except Exception as e:
        print("Failed to get duration:", e)
        return None


def load_mp4_audio_bytes_librosa(mp4_bytes, sr=22050, mono=True):
    """
    Load MP4 audio bytes into librosa using ffmpeg for decoding.
    """
    try:
        # Run ffmpeg to decode mp4 bytes into wav (PCM 16-bit)
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", "pipe:0",        # input from stdin
            "-f", "wav",           # output format
            "pipe:1"               # output to stdout
        ]
        proc = subprocess.run(
            ffmpeg_cmd,
            input=mp4_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        wav_bytes = proc.stdout
        # Load the decoded wav bytes with librosa
        audio_buffer = io.BytesIO(wav_bytes)
        y, sr = librosa.load(audio_buffer, sr=sr, mono=mono)
        # print(f"Audio loaded: {y.shape[0]} samples at {sr} Hz")
        return y, sr
    except subprocess.CalledProcessError as e:
        print("ffmpeg error:", e.stderr.decode())
        return None, None
    except Exception as e:
        print("Failed to load audio with librosa:", e)
        return None, None


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    return default_collate(
        [
            preprocess_conversation(conversation, tokenizer, no_system_prompt=no_system_prompt)
            for conversation in sources
        ]
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and modified by
    Ji Lin and Haotian Tang.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        if os.path.isdir(data_path):
            list_data_dict=[]
            for fname in sorted(os.listdir(data_path)):
                if fname.endswith(".ndjson") or fname.endswith(".jsonl"):
                    full_path = os.path.join(data_path, fname)
                    with open(full_path, "r", encoding="utf-8") as fp:
                        for line in fp:
                            line = line.strip()
                            if line:
                                list_data_dict.append(json.loads(line))
        else:
            try:
                with open(data_path) as fp:
                    list_data_dict = json.load(fp)
            except:
                with open(data_path) as fp:
                    list_data_dict = [json.loads(q) for q in fp]

        # rank0_print("Formatting inputs...Skip in lazy mode")
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder
        self.wav_processor = AutoFeatureExtractor.from_pretrained('Qwen/Qwen2-Audio-7B')
        self.tar_handles = {}
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            if 'duration' in sample.keys():
                duration = sample["duration"]
            else:
                duration = 10.
            try:
                cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"]) + int(math.ceil(duration * 25))
                cur_len = cur_len if "sound" in sample else -cur_len
                length_list.append(cur_len)
            except:
                try:
                    cur_len = 0 + int(math.ceil(duration * 25))
                    cur_len = cur_len if "sound" in sample else -cur_len
                    length_list.append(cur_len)  
                except:
                    cur_len = 0 + int(math.ceil(10. * 25))
                    cur_len = cur_len if "sound" in sample else -cur_len
                    length_list.append(cur_len) 
        return length_list
    

    @staticmethod
    def _load_sound_tar(audio_metadata,audio_dataset_name, tar_handles, wav_processor, sample_rate=16000, window_length=30.0, window_overlap=0.0, max_num_window=3, audio_start = 0.0):
        
        window_length  = int(window_length * sample_rate)
        window_overlap = int(window_overlap * sample_rate)
        max_num_window = int(max_num_window)
        duration = max_num_window * (window_length - window_overlap) + window_overlap
        absolute_path_to_tarball = '/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_audio/users/arushig/datasets/tarred/audio_flamingo'
        sound_outputs = []
        audio_feature_masks = []
        audio_embed_masks = []
        audio_times =[]
        # try:
        audio_data, tar_handles = _load_tarball_local_audio_byteseek(absolute_path_to_tarball, audio_metadata, tar_handles,audio_dataset_name)
        # print(f"audio_data: {audio_data.shape}")
        T = len(audio_data)
            
        audio_data = audio_data.reshape(1, -1)
        num_windows, full_length = get_num_windows(T, sample_rate, max_num_window)

        audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
        count = 0
        for i in range(num_windows):
            audio_time= torch.zeros(750)
            audio_embed_mask = torch.zeros(750)
            start = i * (window_length - window_overlap)
            audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
            orig_length = audio_data_tensor_this.shape[1]
            audio_data_tensor_this = wav_processor(audio_data_tensor_this.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt") #.squeeze(0) text="dummy", audios=audio_data_tensor_this, return_tensors="pt") #
            sound_outputs.append(audio_data_tensor_this["input_features"])
            # calculate the mask for the input melspec to Whisper
            melspec_frames_this_window = int(math.ceil(orig_length / 160))
            feature_attention_mask = torch.zeros(3000, dtype=torch.int32)
            feature_attention_mask[:melspec_frames_this_window] = 1
            audio_feature_masks.append(feature_attention_mask.unsqueeze(0))
            # calculate the mask for the output embedding for use in AF2
            conv_lengths = (melspec_frames_this_window - 1) // 2 + 1
            output_embedding_lengths = (conv_lengths - 2) // 2 + 1
            audio_embed_mask[:output_embedding_lengths] = 1
            audio_embed_masks.append(audio_embed_mask)
            time = torch.arange(count * 30, (count+1) * 0.04 * 750, 0.04)
            audio_time[:750] = time[:750]
            audio_times.append(audio_time)
            count = count + 1
        # except:
        #     print('error loading file', sound_file)
        #     sound_outputs.append(torch.zeros(1,128,3000))
        #     audio_feature_masks.append(torch.zeros(1,3000, dtype=torch.int32))
        #     audio_embed_masks.append(torch.zeros(750))
        #     audio_times.append(torch.zeros(750))

        return torch.stack(sound_outputs, dim=0), torch.stack(audio_feature_masks, dim=0), torch.stack(audio_embed_masks, dim=0), torch.stack(audio_times, dim=0), tar_handles


    @staticmethod
    def _load_sound(sound_file, wav_processor, sample_rate=16000, window_length=30.0, window_overlap=0.0, max_num_window=3, audio_start = 0.0):
        if sound_file is None:
            return None
        window_length  = int(window_length * sample_rate)
        window_overlap = int(window_overlap * sample_rate)
        max_num_window = int(max_num_window)
        duration = max_num_window * (window_length - window_overlap) + window_overlap

        sound_outputs = []
        audio_feature_masks = []
        audio_embed_masks = []
        audio_times =[]
        # try:
        if 's3://' in sound_file:
            # Remove the s3:// prefix and split into bucket and key
            path = sound_file.replace("s3://", "")
            parts = path.split("/", 1)
            bucket_name = parts[0]
            key = parts[1]
            audio_data_bytes = read_audio_from_ss(bucket_name, key)
            audio_data, sr = load_mp4_audio_bytes_librosa(audio_data_bytes, sr=16000)
        else:
            sound_filename = str.split(sound_file, '/')[-1]
            if '.ark' in sound_filename:
                sound = kaldiio.load_mat(sound_file)
                audio_data = sound[1]
                audio_data=audio_data.astype(np.float16)
            # elif '.mp4' in sound_filename:
            #     # Load audio from video file
            #     ar = AudioReader(sound_file, ctx=cpu(0), sample_rate=sample_rate, mono=True)
            #     cur_max_length = ar.shape[1]
            #     audio_data = ar[0:cur_max_length].asnumpy()[0]  # Load the first CHUNK_LENGTH seconds
            # elif '.m4a' in sound_filename:
            #     # Load audio from video file
            #     # ar = AudioReader(sound_file, ctx=cpu(0), sample_rate=sample_rate, mono=True)
            #     # cur_max_length = ar.shape[1]
            #     # audio_data = ar[0:cur_max_length].asnumpy()[0]
            #     audio = AudioSegment.from_file(sound_file, format="m4a")

            #     # Export to bytes
            #     buffer = io.BytesIO()
            #     audio.export(buffer, format="wav")
            #     buffer.seek(0)

            #     audio_data, _ = librosa.load(buffer, sr=16000)
            else:
                audio_data = load_audio(sound_file, sample_rate, duration, audio_start) # already cuts to max duration
        T = len(audio_data)
            
        audio_data = audio_data.reshape(1, -1)
        num_windows, full_length = get_num_windows(T, sample_rate, max_num_window)

        audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
        count = 0
        for i in range(num_windows):
            audio_time= torch.zeros(750)
            audio_embed_mask = torch.zeros(750)
            start = i * (window_length - window_overlap)
            audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
            orig_length = audio_data_tensor_this.shape[1]
            audio_data_tensor_this = wav_processor(audio_data_tensor_this.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt") #.squeeze(0) text="dummy", audios=audio_data_tensor_this, return_tensors="pt") #
            sound_outputs.append(audio_data_tensor_this["input_features"])
            # calculate the mask for the input melspec to Whisper
            melspec_frames_this_window = int(math.ceil(orig_length / 160))
            feature_attention_mask = torch.zeros(3000, dtype=torch.int32)
            feature_attention_mask[:melspec_frames_this_window] = 1
            audio_feature_masks.append(feature_attention_mask.unsqueeze(0))
            # calculate the mask for the output embedding for use in AF2
            conv_lengths = (melspec_frames_this_window - 1) // 2 + 1
            output_embedding_lengths = (conv_lengths - 2) // 2 + 1
            audio_embed_mask[:output_embedding_lengths] = 1
            audio_embed_masks.append(audio_embed_mask)
            time = torch.arange(count * 30, (count+1) * 0.04 * 750, 0.04)
            audio_time[:750] = time[:750]
            audio_times.append(audio_time)
            count = count + 1
        # except:
        #     print('error loading file', sound_file)
        #     sound_outputs.append(torch.zeros(1,128,3000))
        #     audio_feature_masks.append(torch.zeros(1,3000, dtype=torch.int32))
        #     audio_embed_masks.append(torch.zeros(750))
        #     audio_times.append(torch.zeros(750))

        return torch.stack(sound_outputs, dim=0), torch.stack(audio_feature_masks, dim=0), torch.stack(audio_embed_masks, dim=0), torch.stack(audio_times, dim=0)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        entry = self.list_data_dict[i]
        try:
            if "sound" in entry or "audio" in entry:
                # arushig: not modifying the multi turn chat code for not to support tar format
                sound_path = entry["sound"] if "sound" in entry else entry["audio"]['path']
                if isinstance(sound_path, list):
                    sound_files = entry["sound"]                   # e.g. ["a.wav","b.wav",...]
                    conversations_raw = entry["conversations"]     # list of {"from","value"}
                    # 1) Collect tag occurrences in textual order (e.g., <sound-1>, <sound-2>, …)
                    tag_re = re.compile(r"<sound-(\d+)>")
                    ordered_tags = []   # e.g., ["<sound-2>", "<sound-1>", "<sound-2>", ...]
                    for turn in conversations_raw:
                        for m in tag_re.findall(turn["value"]):
                            ordered_tags.append(f"<sound-{m}>")

                    # 2) Load each referenced file once; cache by index k (1-based tag)
                    #    Assumption: sound_files[k-1] corresponds to <sound-k>
                    cache = {}  # k -> (windows_tensor [W, ...], feat_mask [W,...], embed_mask [W,...])
                    def _get_cached(k: int):
                        if k in cache:
                            return cache[k]
                        if k < 1 or k > len(sound_files):
                            raise ValueError(f"Tag <sound-{k}> refers to missing sound file at index {k-1}.")
                        snd_path = sound_files[k-1]
                        # your loader returns (windows, feat_mask, embed_mask); windows is often [W, 1, 750, 2048] or [W, 750, 2048]
                        win, af_mask, ae_mask,a_time = self._load_sound(snd_path, self.wav_processor,
                                                                max_num_window=self.data_args.audio_frames)
                        # match your single-audio behavior: squeeze the extra batch dim if present
                        # (keeps shape [W, 750, 2048] or [W, 1, 750, 2048] depending on your SoundTower path)
                        win = win.squeeze(1)
                        cache[k] = (win, af_mask, ae_mask, a_time)
                        return cache[k]

                    # 3) Build replacement strings and flatten windows/masks in the exact tag order
                    sound_tensors = []         # list of [Wk, ...] to be cat later
                    sound_feat_masks = []      # list of [Wk, ...]
                    sound_embed_masks = []     # list of [Wk, ...]
                    sound_times= []
                    token_map = {}             # "<sound-k>" -> "<sound>\n" * Wk  (for replacement)
                    for tag in ordered_tags:
                        k = int(tag.split("-")[1][:-1])  # "<sound-12>" -> 12
                        win, af_mask, ae_mask, a_time = _get_cached(k)
                        # append these windows for this *occurrence* of the tag
                        sound_tensors.append(win)                   # [Wk, ...]
                        sound_feat_masks.append(af_mask)            # [Wk, ...]
                        sound_embed_masks.append(ae_mask)           # [Wk, ...]
                        sound_times.append(a_time)
                        # remember replacement string for this tag (<sound> repeated Wk times, newline after each)
                        if tag not in token_map:
                            Wk = win.shape[0]
                            token_map[tag] = ("<sound>\n" * Wk).rstrip()

                    # 4) Replace <sound-k> with repeated "<sound>\n" in the conversation
                    conversation = []
                    for turn in conversations_raw:
                        role, value = turn["from"], turn["value"]
                        for tag, marker in token_map.items():
                            value = value.replace(tag, marker)
                        conversation.append({"from": role, "value": value.rstrip()})

                    # 5) Finalize sources (chat) + stack media
                    sources = [conversation]
                    # If no tags were present, fall back to text-only like below.
                    if len(sound_tensors) > 0:
                        sound_tensor = torch.cat(sound_tensors, dim=0)
                        audio_feature_masks = torch.cat(sound_feat_masks, dim=0)
                        audio_embed_masks = torch.cat(sound_embed_masks, dim=0)
                        audio_times = torch.cat(sound_times, dim=0)
                    else:

                        sound_tensor = None
                        audio_feature_masks = None
                        audio_embed_masks = None
                        audio_times = None

                # single turn loading
                elif isinstance(sound_path, str):
                    if "sound" in self.list_data_dict[i]:
                        
                        sound_file = self.list_data_dict[i]["sound"]
                        # print(sound_file)
                        if 'duration' in self.list_data_dict[i]:
                            duration = self.list_data_dict[i]["duration"]
                            # print(duration)
                        else:
                            if ".mat" in sound_file or ".ark" in sound_file:
                                duration = 30.
                            else:
                                try:
                                    duration = librosa.get_duration(path=sound_file)
                                except:
                                    print(f'error getting duration for {sound_file}')
                                    duration = 30.
                        if duration > 1000.:
                            print(f'duration is long: {duration}')
                            # logger.exception(f"Error processing instance '{self.list_data_dict[i]}':. Resampling.")
                            return self.__getitem__(random.randint(0, len(self.list_data_dict) - 1))
                        else:
                            question = str(self.list_data_dict[i]["conversations"][0]["value"].rstrip())
                            answer = str(self.list_data_dict[i]["conversations"][1]["value"]).rstrip()
                            question = question.replace("<speech>\n", "").replace("\n<speech>", "").replace("<speech>", "")
                            question = question.replace("<sound>\n", "").replace("\n<sound>", "").replace("<sound>", "")
                            question = question.replace("<en><asr>\n", "").replace("\n<en><asr>", "").replace("<en><asr>", "")
                            question = question.replace("<eng><asr>\n", "").replace("\n<eng><asr>", "").replace("<eng><asr>", "")
                            sound_tensor, audio_feature_masks, audio_embed_masks, audio_times = self._load_sound(sound_file, self.wav_processor, max_num_window=self.data_args.audio_frames)
                            sound_tensor=sound_tensor.squeeze(1) # squeeze the irrelevant dimension which was caused due to processor getting 1 batch for processing --> (windows x 750 x 2048)

                            if random.choice([True, False]):
                                question = "<sound>" * sound_tensor.shape[0] + "\n" + question
                            else:
                                question = question + "\n" + "<sound>" * sound_tensor.shape[0]
                            conversation = [
                                {"from": "human", "value": question},
                                {"from": "gpt", "value": answer},
                            ]

                            sources = [conversation]
                    else:
                        audio_metadata = self.list_data_dict[i]["audio"]
                        # print(audio_metadata)
                        audio_dataset_name = self.list_data_dict[i]["dataset"]
                        try:
                            duration = self.list_data_dict[i]["audio"]['duration']
                        except:
                            print("no duration found")
                            duration = 30.
                        if duration > 1000.0:
                            print(f'duration is long: {duration}')
                            return self.__getitem__(random.randint(0, len(self.list_data_dict) - 1))
                        else:
                            question = str(self.list_data_dict[i]["conversations"][0]["value"].rstrip())
                            answer = str(self.list_data_dict[i]["conversations"][1]["value"]).rstrip()
                            question = question.replace("<speech>\n", "").replace("\n<speech>", "").replace("<speech>", "")
                            question = question.replace("<sound>\n", "").replace("\n<sound>", "").replace("<sound>", "")
                            question = question.replace("<en><asr>\n", "").replace("\n<en><asr>", "").replace("<en><asr>", "")
                            question = question.replace("<eng><asr>\n", "").replace("\n<eng><asr>", "").replace("<eng><asr>", "")
                            sound_tensor, audio_feature_masks, audio_embed_masks, audio_times, tar_handles = self._load_sound_tar(audio_metadata, audio_dataset_name, self.tar_handles, self.wav_processor, max_num_window=self.data_args.audio_frames)
                            sound_tensor=sound_tensor.squeeze(1) # squeeze the irrelevant dimension which was caused due to processor getting 1 batch for processing --> (windows x 750 x 2048)
                            self.tar_handles = tar_handles
                            if random.choice([True, False]):
                                question = "<sound>" * sound_tensor.shape[0] + "\n" + question
                            else:
                                question = question + "\n" + "<sound>" * sound_tensor.shape[0]
                            conversation = [
                                {"from": "human", "value": question},
                                {"from": "gpt", "value": answer},
                            ]

                            sources = [conversation]
                # text-only data loading 
                else:
                    question = str(self.list_data_dict[i]["conversations"][0]["value"].rstrip())
                    answer = str(self.list_data_dict[i]["conversations"][1]["value"]).rstrip()
                    conversation = [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": answer},
                    ]

                    sources = [conversation]
        except:
            print('error loading file', sound_path)
            # logger.exception(f"Error processing instance '{self.list_data_dict[i]}':. Resampling.")
            return self.__getitem__(random.randint(0, len(self.list_data_dict) - 1))
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "sound" in self.list_data_dict[i] or "audio" in self.list_data_dict[i]
            ),
        )
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        if (len(data_dict["input_ids"]) + len(data_dict["labels"])) > 24000:
            print(f'length of input data long: {len(data_dict["input_ids"]) + len(data_dict["labels"])}')
            # logger.exception(f"Error processing instance: Resampling. data_path: {self.data_path}")
            return self.__getitem__(random.randint(0, len(self.list_data_dict) - 1))
        if "sound" in self.list_data_dict[i] or "audio" in self.list_data_dict[i]:
            data_dict["sound"] = sound_tensor
            data_dict["sound_feature_masks"] = audio_feature_masks
            data_dict["sound_embed_masks"] = audio_embed_masks
            data_dict["sound_times"] = audio_times
        else:
            data_dict["sound"] = None
            data_dict["sound_feature_masks"] = None
            data_dict["sound_embed_masks"] = None
            data_dict["sound_times"] = None
      
        return data_dict


class LazyMMC4Dataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Haotian Tang."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        image_following_text_only=False,
        text_only=False,
    ):
        super().__init__()

        import pickle

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        # n_shards = 100
        count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
        n_samples = [int(open(os.path.join(data_path, f)).read().strip()) for f in count_info_list]

        print("total MMC4 samples", sum(n_samples))  # 10,881,869

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                data_list = pickle.load(f)

            full_data_list.extend(data_list)

        print(f"* loaded totally {len(full_data_list)} samples")

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

        self.image_following_text_only = image_following_text_only
        self.text_only = text_only

    def __len__(self):
        # return len(self.data_list)
        return self.n_samples

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for info in self.data_list:
            num_images = min(6, len(info["image_info"]))
            sentences = [info["text_list"][x["matched_text_index"]] for x in info["image_info"][:num_images]]
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = num_images * self.num_image_tokens // 2 + sum([len(x) for x in sentences])
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        info = self.data_list[i - self.idx_offset]

        sentences = info["text_list"]
        # kentang-mit@: remove existing <image> tokens in the sentences
        for ix in range(len(sentences)):
            # if this is an html tag, we still preserve its semantic meaning
            sentences[ix] = sentences[ix].replace("<image>", "<IMAGE>")
        sim_matrix = info["similarity_matrix"]  # we do not use this...

        # convert images from base64 to PIL and filter based on image-text similarity
        images, sentence_ixs = [], []
        if not self.text_only:
            for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
                image_base64 = sample_image["image_base64"]
                rawbytes = base64.b64decode(image_base64)

                sim_ix = sample_image["matched_text_index"]
                # sim_ix = np.argmax(sim_vec)
                # sim_score = sim_vec[sim_ix]

                # filter to images >= 5KB
                # if len(rawbytes) // 1000 <= 5:
                #     continue
                # if sim_score < 0.24:
                #     continue
                image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

                images.append(image)
                sentence_ixs.append(sim_ix)

        # constrain max num 6 images
        max_num_images = 6
        if len(images) > max_num_images:
            images = images[:max_num_images]
            sentence_ixs = sentence_ixs[:max_num_images]

        # reorder images according to text insertion
        images = [images[iii] for iii in np.argsort(sentence_ixs)]

        # preprocess and tokenize text
        for ix in sentence_ixs:
            sentences[ix] = f"<image>\n{sentences[ix]}"

        if self.image_following_text_only:
            # use pad tokens to divide sentence pieces
            text = self.tokenizer.pad_token.join(sentences)
        else:
            text = " ".join(sentences)
        # whitespace cleanup
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        text = f"{text}{self.tokenizer.eos_token}"  # add eos token

        if len(images) > 0:
            if self.data_args.image_aspect_ratio == "dynamic_s2":
                images, block_sizes = dynamic_s2_process_images_and_prompt(
                    images, text, self.data_args, self.image_folder
                )
            elif self.data_args.image_aspect_ratio == "dynamic":
                images, text = dynamic_process_images_and_prompt(
                    images, text, self.data_args, self.image_folder, max_tiles=6
                )
            else:
                images = torch.stack([process_image(image, self.data_args, self.image_folder) for image in images])

            # the same size for all images, so we concat
            # cur_token_len = (
            #     images[0].shape[-2] // self.multimodal_cfg["patch_size"]
            # ) * (images[0].shape[-1] // self.multimodal_cfg["patch_size"])
            # cur_token_len += self.multimodal_cfg["n_extra_patch"]
        else:
            images = None
            # cur_token_len = 0

        input_ids = tokenizer_image_token(
            text,
            self.tokenizer,
            return_tensors="pt",
        )

        image_token_id = self.tokenizer.media_token_ids["image"]

        # now check the case where the last token is image patch token
        if input_ids[-1] == image_token_id:  # need to remove one last image
            last_non_im_patch_indices = torch.where(input_ids != image_token_id)[0][-1] + 1
            input_ids = input_ids[:last_non_im_patch_indices]

        n_im_patch = (input_ids == image_token_id).sum().item()

        if self.data_args.image_aspect_ratio != "dynamic_s2":
            images = images[:n_im_patch]
            assert len(images) == n_im_patch, print(text, input_ids)
        assert len(input_ids.shape) == 1, "Unexpected shape of 'input_ids' from MMC4."
        input_ids = (
            torch.concat([torch.tensor([self.tokenizer.bos_token_id]), input_ids])
            if self.tokenizer.bos_token_id is not None and input_ids[0] != self.tokenizer.bos_token_id
            else input_ids
        )
        targets = input_ids.clone()

        if self.image_following_text_only:  # keep only text after leading image token
            # remove loss for any token before the first <image> token
            label_idx = 0
            while label_idx < targets.shape[-1] and targets[label_idx] != image_token_id:
                targets[label_idx] = IGNORE_INDEX
                label_idx += 1

            pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

            pad_token_idxs = torch.where(targets == pad_token)[0]
            for pad_token_idx in pad_token_idxs:
                token_idx = pad_token_idx + 1
                while token_idx < targets.shape[-1] and targets[token_idx] != image_token_id:
                    targets[token_idx] = IGNORE_INDEX
                    token_idx += 1
            # do not train on padding tokens
            targets[targets == pad_token] = IGNORE_INDEX

        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        # print(input_ids.shape)

        data_dict = dict(input_ids=input_ids, labels=targets, image=images)
        if self.data_args.image_aspect_ratio == "dynamic_s2":
            data_dict["block_sizes"] = block_sizes

        return data_dict


class LazyCoyoDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Haotian Tang."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # kentang-mit@: balance the total number of tokens for Coyo and MMC4.
        n_samples_per_idx=4,
    ):
        super().__init__()

        import pickle

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        # n_shards = 100
        count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
        n_samples = [int(open(os.path.join(data_path, f)).read().strip()) for f in count_info_list]

        print("total COYO samples", sum(n_samples))

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [
            sum(n_samples[i * shared_size : (i + 1) * shared_size]) // n_samples_per_idx for i in range(world_size)
        ]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)

        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                shard_data = pickle.load(f)
                random.seed(42)
                if "mmc4" in data_path:
                    random.shuffle(shard_data)  # shuffle for MMC4cap only
                full_data_list.extend(shard_data)

        print(f"* loaded totally {len(full_data_list)} samples")

        # now pack the samples into groups
        n_groups = len(full_data_list) // n_samples_per_idx
        full_data_list = [
            full_data_list[i : i + n_samples_per_idx] for i in range(0, len(full_data_list), n_samples_per_idx)
        ]
        if len(full_data_list[-1]) < n_samples_per_idx:
            full_data_list = full_data_list[:-1]
        assert len(full_data_list) == n_groups
        print(f"split into {n_groups} groups")

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        # return len(self.data_list)
        return self.n_samples

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        info_list = self.data_list[i - self.idx_offset]

        text_list = []
        image_list = []

        for sample in info_list:
            caption_key = (
                "text" if "text" in sample else "caption"
            )  # kentang-mit@: remove existing <image> tokens in the sentences
            # kentang-mit@: remove existing <image> token.
            # if this is an html tag, we still preserve its semantic meaning
            sample[caption_key] = sample[caption_key].replace("<image>", "<IMAGE>")
            text_list.append(DEFAULT_IMAGE_TOKEN + "\n" + sample[caption_key] + self.tokenizer.eos_token)
            if "image" in sample:
                image_base64 = sample["image"]
                rawbytes = base64.b64decode(image_base64)
            else:
                rawbytes = sample["rawbytes"]
            image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
            image_list.append(image)

        image_list = torch.stack([process_image(image, self.data_args, self.image_folder) for image in image_list])

        if CONCAT_SAMPLES:
            # into <image>cap<eos><image>cap<eos>...
            text_list = "".join(text_list)

            input_ids = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids  # 4, seq_len

            input_ids = input_ids[0]

        else:
            input_ids = [
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    return_tensors="pt",
                )
                for prompt in text_list
            ]
            # print([x.shape[0] for x in input_ids], [len(x.split()) for x in text_list], [len(re.findall(r"<image[^>]*>", x)) for x in text_list])

            # input_ids = torch.nn.utils.rnn.pad_sequence(
            #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            # )

        targets = copy.deepcopy(input_ids)
        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=image_list)


class LazyWDSDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Ligeng Zhu."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        image_folder: str,
        training_args: TrainingArguments,
    ):
        super().__init__()
        n_samples = []
        n_shards = len(os.listdir(data_path)) // 3
        for shard in range(n_shards):
            with open(os.path.join(data_path, f"{shard:05d}_stats.json")) as f:
                info = json.load(f)
                n_samples.append(info["successes"])

        # print(f"[DEBUG] {data_path} total samples", sum(n_samples))  # 10,881,869

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size
        print("rank", rank, "world_size", world_size, "shared_size", shared_size)
        gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        tar_list = [f"{shard_idx:05d}.tar" for shard_idx in range(shard_start, shard_end)]

        self.data_list = []
        t1 = time.time()
        for tar in tar_list:
            tmp_path = f"/tmp/ccs{tar}"
            tar_path = os.path.join(data_path, tar)

            if PROCESS_GROUP_MANAGER is not None:
                dist.barrier()
                if PROCESS_GROUP_MANAGER.sp_rank == 0:
                    os.makedirs(tmp_path, exist_ok=True)
                    os.system(f"tar -xkf {tar_path} -C {tmp_path}")
                dist.barrier()
            else:
                os.makedirs(tmp_path, exist_ok=True)
                os.system(f"tar -xkf {tar_path} -C {tmp_path}")

            txt_list = [f for f in os.listdir(tmp_path) if f.endswith(".txt")]

            for txt in txt_list:
                caption = open(os.path.join(tmp_path, txt)).read().strip()
                image_path = os.path.join(tmp_path, txt.split(".")[0] + ".jpg")
                self.data_list.append({"caption": caption, "image": image_path})
        t2 = time.time()
        print(f"Loading done. Total time: {t2 - t1:.2f} seconds")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        # print("i", i, "idx_offset", self.idx_offset, "len", len(self.data_list))
        info = self.data_list[i - self.idx_offset]
        caption, image_path = info["caption"], info["image"]

        rand_prompt = "<image>\n"
        sources = [
            {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": rand_prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        ]

        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        if "image" in sources[0]:
            image = process_image(sources[0]["image"], self.data_args, self.image_folder)
            image = torch.unsqueeze(image, dim=0)
            # now random pick some context samples for training
            if hasattr(self.data_args, "num_shots"):
                if self.data_args.num_shots > 0:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        data_dict = preprocess([sources[0]["conversations"]], self.tokenizer, has_image=True)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if image is not None:
            data_dict["image"] = image
        else:
            raise NotImplementedError

        return data_dict


class LazyCCSWebDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ligeng Zhu."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        t1 = time.time()

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(data_path=osp.abspath(data_path))

        t2 = time.time()
        print(f"Loading done. Total time: {t2 - t1:.2f} seconds")

        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # info = self.data_list[i - self.idx_offset]
        # caption, image_path = info["caption"], info["image"]
        info = self.dataset[i]
        if ".jpg" in info:
            caption, image_path = info[".txt"], info[".jpg"]
        elif ".png" in info:
            caption, image_path = info[".txt"], info[".png"]
        elif ".webp" in info:
            caption, image_path = info[".txt"], info[".webp"]
        elif ".bmp" in info:
            caption, image_path = info[".txt"], info[".bmp"]
        elif ".tiff" in info:
            caption, image_path = info[".txt"], info[".tiff"]
        else:
            print(info.keys())
            print(info)
            raise KeyError

        caption = caption.replace("<image>", "<IMAGE>")
        if isinstance(image_path, io.BytesIO):
            image_path = Image.open(image_path).convert("RGB")

        if not isinstance(image_path, PIL.Image.Image):
            print(image_path)
            print(info.keys())
            print(type(image_path))
            raise NotImplementedError

        rand_prompt = "<image>\n"
        sources = [
            {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": rand_prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        ]

        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        if "image" in sources[0]:
            image = process_image(sources[0]["image"], self.data_args, image_folder=None)
            image = torch.unsqueeze(image, dim=0)
            # now random pick some context samples for training
            if hasattr(self.data_args, "num_shots"):
                if self.data_args.num_shots > 0:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        data_dict = preprocess([sources[0]["conversations"]], self.tokenizer, has_image=True)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if image is not None:
            data_dict["image"] = image
        else:
            raise NotImplementedError

        return data_dict


from functools import lru_cache


@lru_cache(maxsize=16)
def lru_json_load(fpath):
    with open(fpath) as fp:
        return json.load(fp)


class LazyCoyoWebDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ligeng Zhu."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # kentang-mit@: balance the total number of tokens for Coyo and MMC4.
        n_samples_per_idx=4,
    ):
        super().__init__()

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(data_path=osp.abspath(data_path), meta_path=data_args.meta_path)

        if data_args.start_idx >= 0 and data_args.end_idx >= 0:
            # Ligeng: support slicing for ablate different subsets.
            total = len(self.dataset)
            start_idx = int(total * data_args.start_idx)
            end_idx = int(total * data_args.end_idx)
            print(f"loading subset from {start_idx} to {end_idx}, total {total}")
            self.dataset = torch.utils.data.Subset(self.dataset, range(start_idx, end_idx))

        # For caption choice,
        #   if None: use original caption
        #   if a folder path: use specified caption to override original one (choice1)
        #   if a folder path: use specified caption and concat with original one (choice2)
        self.caption_choice = None
        self.caption_choice_2 = None
        self.data_path = data_path

        if data_args.caption_choice is not None:
            self.caption_choice = data_args.caption_choice
            print("[recap] Override coyo caption using ", self.caption_choice)

        if data_args.caption_choice_2 is not None:
            self.caption_choice_2 = data_args.caption_choice_2
            print("[recapv2] Override coyo caption using ", self.caption_choice_2)

        print("total samples", len(self.dataset))
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
            sequence_parallel_rank = PROCESS_GROUP_MANAGER.sp_rank
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = (
            training_args.process_index // sequence_parallel_size if "RANK" in os.environ else 2
        )  # int(os.environ["RANK"])
        world_size = (
            training_args.world_size // sequence_parallel_size if "WORLD_SIZE" in os.environ else 32
        )  # int(os.environ["WORLD_SIZE"])
        print(
            "rank",
            rank,
            "world_size",
            world_size,
        )

        self.n_samples_per_idx = n_samples_per_idx
        # self.n_samples = len(self.dataset) // n_samples_per_idx
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset) // self.n_samples_per_idx

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        # info_list = self.dataset[i - self.idx_offset]

        begin_idx, end_idx = (
            i * self.n_samples_per_idx,
            (i + 1) * self.n_samples_per_idx,
        )
        end_idx = min(end_idx, len(self.dataset))

        text_list = []
        image_list = []

        for idx in range(begin_idx, end_idx):
            info = self.dataset[idx]
            if ".jpg" in info:
                caption, image_path = info[".txt"], info[".jpg"]
            elif ".png" in info:
                caption, image_path = info[".txt"], info[".png"]
            elif ".webp" in info:
                caption, image_path = info[".txt"], info[".webp"]
            elif ".bmp" in info:
                caption, image_path = info[".txt"], info[".bmp"]
            elif ".tiff" in info:
                caption, image_path = info[".txt"], info[".tiff"]
            else:
                print(info.keys())
                print(info)
                raise KeyError

            if self.caption_choice is not None:
                # load new captions
                shard = info["__shard__"]
                url = info[".json"]["url"]
                tar_name = osp.relpath(osp.realpath(shard), osp.realpath(self.data_path))
                # tar_name = osp.dirname(shard)
                shard_json_path = osp.join(self.caption_choice, tar_name + ".json")
                try:
                    shard_json = lru_json_load(shard_json_path)
                    try:
                        caption = shard_json[url]["output"]
                    except KeyError:
                        print(f"{url} not in caption. fallback to original caption temporarially")
                except:
                    print(f"shard_json_path {shard_json_path} not found. fallback to original caption temporarially")
            caption = caption.replace("<image>", "<IMAGE>")
            text_list.append(DEFAULT_IMAGE_TOKEN + caption + self.tokenizer.eos_token)

            if isinstance(image_path, io.BytesIO):
                image_path = Image.open(image_path).convert("RGB")

            if not isinstance(image_path, PIL.Image.Image):
                print(image_path)
                print(info.keys())
                print(type(image_path))
                raise NotImplementedError

            image_list.append(image_path)

        # image_list = torch.stack([process_image(image, self.data_args, image_folder=None) for image in image_list])
        # NOTE(fix by ligeng)
        #  now image_list should return a list of image tensor where each has a dimension of (1, c, h, w)
        image_list = [process_image(image, self.data_args, image_folder=None).unsqueeze(0) for image in image_list]

        if CONCAT_SAMPLES:
            # into <image>cap<eos><image>cap<eos>...
            text_list = "".join(text_list)

            input_ids = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids  # 4, seq_len

            input_ids = input_ids[0]
        else:
            input_ids = [
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    return_tensors="pt",
                )
                for prompt in text_list
            ]
            input_ids = [
                (
                    torch.concat([torch.tensor([self.tokenizer.bos_token_id]), input_ids_i])
                    if input_ids_i[0] != self.tokenizer.bos_token_id
                    else input_ids_i
                )
                for input_ids_i in input_ids
            ]

        targets = copy.deepcopy(input_ids)
        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=image_list)


class LazyVideoWebDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # cache_path: str,
        # n_samples_per_idx=4,
    ):
        super().__init__()

        # from llava.data.simple_video_dataset import SimpleVideoDataset

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(
            data_path=osp.abspath(data_path),
            meta_path=f"{osp.abspath(data_path)}/wids-meta.json",
            # cache_dir=cache_path,
        )

        # None: use original caption
        # Folder path: use original caption
        self.caption_choice = None
        self.data_path = data_path

        if data_args.caption_choice is not None:
            self.caption_choice = data_args.caption_choice
            print("[recap] Override LazyVideo caption using ", self.caption_choice)

        print("total samples", len(self.dataset))
        # InternVid: TODO
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
            sequence_parallel_rank = PROCESS_GROUP_MANAGER.sp_rank
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = (
            training_args.process_index // sequence_parallel_size if "RANK" in os.environ else 2
        )  # int(os.environ["RANK"])
        world_size = (
            training_args.world_size // sequence_parallel_size if "WORLD_SIZE" in os.environ else 32
        )  # int(os.environ["WORLD_SIZE"])
        print(
            "rank",
            rank,
            "world_size",
            world_size,
        )
        self.rank = rank
        # rank = int(os.environ["RANK"]) if "RANK" in os.environ else 2
        # world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 32

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.missing_uids = set()

    def __len__(self):
        return len(self.dataset)

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ADD_TEXT_PROMPT = False
        num_video_frames = self.data_args.num_video_frames if hasattr(self.data_args, "num_video_frames") else 8
        loader_fps = self.data_args.fps if hasattr(self.data_args, "fps") else 0.0

        info = self.dataset[i]

        caption = ""
        # print(info)
        if ".mp4" in info:
            caption, video_path = info[".txt"], info[".mp4"]
        else:
            video_path = None
            caption = "Empty video."

        images, frames_loaded, _ = LazySupervisedDataset._load_video(
            video_path, num_video_frames, loader_fps, self.data_args
        )

        if frames_loaded == 0:
            caption = "Empty video."

        if self.caption_choice is not None:
            shard = info["__shard__"]
            uuid = osp.join(info["__shard__"], info["__key__"])
            url = info["__key__"]
            tar_name = osp.basename(info["__shard__"])

            try:
                shard_json_path = osp.join(self.caption_choice, tar_name.replace(".tar", ".json"))
                shard_json = lru_json_load(shard_json_path)
                caption = shard_json[url]["summary"]["output"]
            except (KeyError, FileNotFoundError, json.decoder.JSONDecodeError):
                if uuid not in self.missing_uids:
                    print("override caption not found for ", uuid)
                    self.missing_uids.add(uuid)

            # print(f"[DEBUG {uuid}]", caption)

        frames_loaded_successfully = len(images)
        if caption is None:
            caption = ""
        prompt = "<image>\n" * frames_loaded_successfully + caption
        image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )
        targets = copy.deepcopy(input_ids)
        data_dict = dict(input_ids=input_ids, labels=targets, image=image_tensor)

        return data_dict


class DataCollatorForSupervisedDatasetSeqParallel:
    """Collate examples for supervised fine-tuning (audio version).
    Adapted from LLaVA sequence-packing collator to support audio inputs instead of images/videos.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        sp_degree: int,
        sp_rank: int,
        ring_degree: int,
        ring_type: str,
    ):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args
        self.sp_degree = sp_degree
        self.sp_rank = sp_rank
        self.ring_degree = ring_degree
        self.ring_type = ring_type

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, audios = [], [], []
        audio_token_id = self.tokenizer.media_token_ids["sound"]

        # --- Step 1: Collect all inputs ---
        for instance in instances:
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]

            if not isinstance(instance["labels"], list):
                labels.append(instance["labels"])
            else:
                labels += instance["labels"]

            # Expect `instance["audio"]` tensor of shape (n_audios, 128, 3000)
            if "sound" in instance:
                cur_audio = instance["sound"]
                assert len(cur_audio.shape) == 3, f"Expected (n_audios, 128, 3000), got {cur_audio.shape}"
                if cur_audio.shape[0] == 0:
                    warnings.warn("Loaded one sample without audio.")
                if not isinstance(instance["input_ids"], list):
                    audios.append(cur_audio)
                else:
                    audios.extend(cur_audio.chunk(cur_audio.size(0), 0))
            else:
                warnings.warn("Loaded one sample without audio.")
                audios.append([])

        # --- Step 2: Sanity checks ---
        max_num_audios = max([len(_a) for _a in audios])
        for _a, _ids in zip(audios, input_ids):
            assert (
                len(_a) == (_ids == audio_token_id).sum().item()
            ), f"Mismatch between number of audio tensors and <audio> tokens. Found {len(_a)} audios but {_ids.tolist().count(audio_token_id)} tokens."

        NUM_TOKENS_PER_AUDIO = getattr(self.data_args, "audio_frames", 10)  # configurable

        # Dummy audio input for padding
        dummy_audio = torch.ones((1, 128, 3000), device=input_ids[0].device)
        dummy_input_ids = torch.tensor(
            [self.tokenizer.bos_token_id, audio_token_id, self.tokenizer.eos_token_id],
            device=input_ids[0].device,
        )
        dummy_labels = dummy_input_ids.clone()
        dummy_labels[:2] = IGNORE_INDEX
        dummy_seqlen = NUM_TOKENS_PER_AUDIO + 2
        dummy_position_ids = torch.arange(0, dummy_seqlen, dtype=torch.int32)

        # --- Step 3: Sort by effective length ---
        combined = sorted(
            zip(input_ids, labels, audios),
            key=lambda x: len(x[2]) * (NUM_TOKENS_PER_AUDIO - 1) + x[0].size(-1),
            reverse=True,
        )
        sorted_ids, sorted_labels, sorted_audios = zip(*combined)
        sorted_ids, sorted_labels, sorted_audios = list(sorted_ids), list(sorted_labels), list(sorted_audios)

        max_seq_length = self.tokenizer.model_max_length
        max_sample_len = 0

        batches, label_batches, position_ids, batch_audios, seqlens_in_batch = [], [], [], [], []
        i = 0

        # --- Step 4: Sequence Packing ---
        while i < len(sorted_ids):
            current_batch = torch.tensor([], dtype=torch.int32)
            current_label_batch = torch.tensor([], dtype=torch.int32)
            current_position_ids = torch.tensor([], dtype=torch.int32)
            current_batch_audios = []
            current_num_audios = 0
            current_len = 0
            current_num_samples = 0

            while i < len(sorted_ids):
                num_audios = (sorted_ids[i] == audio_token_id).sum().item()
                num_audio_tokens_added = num_audios * (NUM_TOKENS_PER_AUDIO - 1)
                num_incoming_tokens = sorted_ids[i].size(-1) + num_audio_tokens_added

                # Handle Ring padding
                if self.ring_degree > 1:
                    RING_PAD_TOKEN_INDEX = 2
                    pad_len = 0
                    if self.ring_type == "ring_varlen":
                        if num_incoming_tokens % self.sp_degree != 0:
                            pad_len = self.sp_degree - num_incoming_tokens % self.sp_degree
                    elif self.ring_type == "zigzag_ring_varlen":
                        zigzag_sp_degree = self.sp_degree * 2
                        if num_incoming_tokens % zigzag_sp_degree != 0:
                            pad_len = zigzag_sp_degree - num_incoming_tokens % zigzag_sp_degree
                    else:
                        raise ValueError(f"Invalid ring_type: {self.ring_type}")

                    if pad_len > 0:
                        pad_tensor = torch.full(
                            (pad_len,), RING_PAD_TOKEN_INDEX, dtype=sorted_ids[i].dtype, device=sorted_ids[i].device
                        )
                        sorted_ids[i] = torch.cat([sorted_ids[i], pad_tensor])
                        pad_label_tensor = torch.full(
                            (pad_len,), IGNORE_INDEX, dtype=sorted_labels[i].dtype, device=sorted_labels[i].device
                        )
                        sorted_labels[i] = torch.cat([sorted_labels[i], pad_label_tensor])
                        num_incoming_tokens += pad_len

                if (current_len + num_incoming_tokens <= max_seq_length):
                    current_num_audios += num_audios
                    current_len += num_incoming_tokens
                    current_num_samples += 1
                    current_position_ids = torch.cat(
                        (current_position_ids, torch.arange(start=0, end=num_incoming_tokens)), dim=0
                    )
                    current_batch = torch.cat((current_batch, sorted_ids[i]), dim=0)
                    sorted_labels[i][0] = IGNORE_INDEX
                    current_label_batch = torch.cat((current_label_batch, sorted_labels[i]), dim=0)
                    seqlens_in_batch.append(num_incoming_tokens)
                    current_batch_audios.extend(sorted_audios[i])
                    i += 1
                    assert current_num_audios == len(current_batch_audios)
                else:
                    break

            # --- Step 5: Padding with dummy audio if needed ---
            MAX_RETRY = self.sp_degree
            num_retry = 0
            while current_num_audios < self.sp_degree and current_len < max_seq_length and num_retry <= MAX_RETRY:
                current_num_audios += dummy_audio.size(0)
                current_len += dummy_seqlen
                current_num_samples += 1
                current_position_ids = torch.cat((current_position_ids, dummy_position_ids), dim=0)
                current_batch = torch.cat((current_batch, dummy_input_ids), dim=0)
                current_label_batch = torch.cat((current_label_batch, dummy_labels), dim=0)
                seqlens_in_batch.append(dummy_seqlen)
                current_batch_audios.extend(dummy_audio)
                num_retry += 1

            if current_num_audios < self.sp_degree:
                print(f"Warning: Skipping one packed sample with {current_num_audios} audios")
                seqlens_in_batch = seqlens_in_batch[:-current_num_samples]
                continue

            max_sample_len = max(max_sample_len, current_len)
            batches.append(current_batch)
            label_batches.append(current_label_batch)
            position_ids.append(current_position_ids)
            batch_audios.append(current_batch_audios)

        # --- Step 6: Sequence parallelism split ---
        for i in range(len(batches)):
            audio_token_indices = torch.where(batches[i] == audio_token_id)[0].tolist()
            audio_ids = torch.arange(0, len(audio_token_indices), dtype=torch.int32)
            batches[i] = extract_local_input_ids(
                batches[i], audio_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
            )
            label_batches[i] = extract_local_input_ids(
                label_batches[i], audio_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
            )
            batch_audios[i] = torch.concat(
                extract_local_from_list(batch_audios[i], self.sp_rank, self.sp_degree), dim=0
            )
            num_audios = len(batch_audios[i])
            assert num_audios == len(torch.where(batches[i] == audio_token_id)[0].tolist())

            position_ids[i] = extract_local_position_ids(
                position_ids[i], audio_token_indices, audio_ids, self.sp_rank, self.sp_degree, NUM_TOKENS_PER_AUDIO - 1
            )

        # --- Step 7: Final tensorization ---
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batches, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(label_batches, batch_first=True, padding_value=IGNORE_INDEX)
        seqlens_in_batch = torch.stack([torch.tensor(x) for x in seqlens_in_batch], axis=0).flatten()
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=-1)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            seqlens_in_batch=seqlens_in_batch,
            media={"sound": batch_audios},
            media_config={"sound": {}},
            position_ids=position_ids,
        )
        return batch


def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.
    This function is originally implemented by the LLaVA team and
    modified by Jason Lu, Haotian Tang and Ligeng Zhu."""
    datasets_mixture.register_datasets_mixtures()

    from .builder import build_dataset

    train_dataset = build_dataset(data_args.data_mixture, data_args, training_args, tokenizer)
    training_args.sample_lens = [len(d) for d in train_dataset.datasets]

    PROCESS_GROUP_MANAGER = get_pg_manager()
    if PROCESS_GROUP_MANAGER is None:
        data_collator = DataCollator(tokenizer=tokenizer)
    else:
        sp_degree = training_args.seq_parallel_size
        sp_rank = PROCESS_GROUP_MANAGER.sp_rank
        ring_degree = PROCESS_GROUP_MANAGER.ring_degree
        ring_type = PROCESS_GROUP_MANAGER.ring_type
        data_collator = DataCollatorForSupervisedDatasetSeqParallel(
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
            sp_degree=sp_degree,
            sp_rank=sp_rank,
            ring_degree=ring_degree,
            ring_type=ring_type,
        )

    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
