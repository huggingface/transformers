# Copyright 2023 The HuggingFace Team. All rights reserved.
import datetime
import platform
import subprocess
from typing import Optional, Tuple, Union

import numpy as np


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


def ffmpeg_microphone(
    sampling_rate: int,
    chunk_length_s: float,
    format_for_conversion: str = "f32le",
):
    """
    Helper function ro read raw microphone data.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    if format_for_conversion == "s16le":
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    system = platform.system()
    if system == "Linux":
        format_ = "alsa"
        input_ = "default"
    elif system == "Darwin":
        format_ = "avfoundation"
        input_ = ":0"
    elif system == "Windows":
        format_ = "dshow"
        input_ = "default"

    ffmpeg_command = [
        "ffmpeg",
        "-f",
        format_,
        "-i",
        input_,
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-fflags",
        "nobuffer",
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    iterator = _ffmpeg_stream(ffmpeg_command, chunk_len)
    for item in iterator:
        yield item


def ffmpeg_microphone_live(
    sampling_rate: int,
    chunk_length_s: float,
    stream_chunk_s: Optional[int] = None,
    stride_length_s: Optional[Union[Tuple[float, float], float]] = None,
    format_for_conversion: str = "f32le",
):
    """
    Helper function to read audio from the microphone file through ffmpeg. This will output `partial` overlapping
    chunks starting from `stream_chunk_s` (if it is defined) until `chunk_length_s` is reached. It will make use of
    striding to avoid errors on the "sides" of the various chunks.

    Arguments:
        sampling_rate (`int`):
            The sampling_rate to use when reading the data from the microphone. Try using the model's sampling_rate to
            avoid resampling later.
        chunk_length_s (`float` or `int`):
            The length of the maximum chunk of audio to be sent returned. This includes the eventual striding.
        stream_chunk_s (`float` or `int`)
            The length of the minimal temporary audio to be returned.
        stride_length_s (`float` or `int` or `(float, float)`, *optional*, defaults to `None`)
            The length of the striding to be used. Stride is used to provide context to a model on the (left, right) of
            an audio sample but without using that part to actually make the prediction. Setting this does not change
            the length of the chunk.
        format_for_conversion (`str`, defalts to `f32le`)
            The name of the format of the audio samples to be returned by ffmpeg. The standard is `f32le`, `s16le`
            could also be used.
    Return:
        A generator yielding dictionaries of the following form

        `{"sampling_rate": int, "raw": np.array(), "partial" bool}` With optionnally a `"stride" (int, int)` key if
        `stride_length_s` is defined.

        `stride` and `raw` are all expressed in `samples`, and `partial` is a boolean saying if the current yield item
        is a whole chunk, or a partial temporary result to be later replaced by another larger chunk.


    """
    if stream_chunk_s is not None:
        chunk_s = stream_chunk_s
    else:
        chunk_s = chunk_length_s

    microphone = ffmpeg_microphone(sampling_rate, chunk_s, format_for_conversion=format_for_conversion)
    if format_for_conversion == "s16le":
        dtype = np.int16
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        dtype = np.float32
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    if isinstance(stride_length_s, (int, float)):
        stride_length_s = [stride_length_s, stride_length_s]

    stride_left = int(round(sampling_rate * stride_length_s[0])) * size_of_sample
    stride_right = int(round(sampling_rate * stride_length_s[1])) * size_of_sample
    audio_time = datetime.datetime.now()
    delta = datetime.timedelta(seconds=chunk_s)
    for item in chunk_bytes_iter(microphone, chunk_len, stride=(stride_left, stride_right), stream=True):
        # Put everything back in numpy scale
        item["raw"] = np.frombuffer(item["raw"], dtype=dtype)
        item["stride"] = (
            item["stride"][0] // size_of_sample,
            item["stride"][1] // size_of_sample,
        )
        item["sampling_rate"] = sampling_rate
        audio_time += delta
        if datetime.datetime.now() > audio_time + 10 * delta:
            # We're late !! SKIP
            continue
        yield item


def chunk_bytes_iter(iterator, chunk_len: int, stride: Tuple[int, int], stream: bool = False):
    """
    Reads raw bytes from an iterator and does chunks of length `chunk_len`. Optionally adds `stride` to each chunks to
    get overlaps. `stream` is used to return partial results even if a full `chunk_len` is not yet available.
    """
    acc = b""
    stride_left, stride_right = stride
    if stride_left + stride_right >= chunk_len:
        raise ValueError(
            f"Stride needs to be strictly smaller than chunk_len: ({stride_left}, {stride_right}) vs {chunk_len}"
        )
    _stride_left = 0
    for raw in iterator:
        acc += raw
        if stream and len(acc) < chunk_len:
            stride = (_stride_left, 0)
            yield {"raw": acc[:chunk_len], "stride": stride, "partial": True}
        else:
            while len(acc) >= chunk_len:
                # We are flushing the accumulator
                stride = (_stride_left, stride_right)
                item = {"raw": acc[:chunk_len], "stride": stride}
                if stream:
                    item["partial"] = False
                yield item
                _stride_left = stride_left
                acc = acc[chunk_len - stride_left - stride_right :]
    # Last chunk
    if len(acc) > stride_left:
        item = {"raw": acc, "stride": (_stride_left, 0)}
        if stream:
            item["partial"] = False
        yield item


def _ffmpeg_stream(ffmpeg_command, buflen: int):
    """
    Internal function to create the generator of data through ffmpeg
    """
    bufsize = 2**24  # 16Mo
    try:
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=bufsize) as ffmpeg_process:
            while True:
                raw = ffmpeg_process.stdout.read(buflen)
                if raw == b"":
                    break
                yield raw
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename") from error
