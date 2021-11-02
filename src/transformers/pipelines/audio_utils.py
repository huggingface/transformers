import collections
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
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except FileNotFoundError:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename")
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]

    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


def ffmpeg_stream(filename: str, sampling_rate: int, format_for_conversion: str, chunk_length_s: float):
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    bufsize = 10 ** 8
    if format_for_conversion == "s16le":
        dtype = np.int16
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        dtype = np.float32
        size_of_sample = 4
    else:
        raise ValueError("Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        filename,
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

    buflen = int(sampling_rate * chunk_length_s * size_of_sample)
    return _ffmpeg_stream(ffmpeg_command, bufsize, buflen, dtype)


def ffmpeg_microphone(
    sampling_rate: int, format_for_conversion: str, chunk_length_s: float, stream_chunk_ms: Optional[int] = None
):
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    bufsize = 10 ** 8
    if format_for_conversion == "s16le":
        dtype = np.int16
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        dtype = np.float32
        size_of_sample = 4
    else:
        raise ValueError("Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

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

    if stream_chunk_ms is not None:
        buflen = int(round(sampling_rate * stream_chunk_ms * size_of_sample / 1000))
        bufout = int(round(sampling_rate * chunk_length_s * size_of_sample))
    else:
        buflen = int(round(sampling_rate * chunk_length_s * size_of_sample))
        bufout = None
    return _ffmpeg_stream(ffmpeg_command, bufsize, buflen, dtype, bufout=bufout)


def _ffmpeg_stream(ffmpeg_command, bufsize, buflen, dtype, bufout=None):
    if bufout is None:
        bufout = buflen
    elif buflen > bufout:
        raise ValueError("bufout needs to be larger than buflen")

    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=-1)
    except FileNotFoundError:
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename")

    acc = b""
    while True:
        raw = ffmpeg_process.stdout.read(buflen)
        if raw == b"":
            break

        if len(acc) + len(raw) > bufout:
            acc = raw
        else:
            acc += raw
        audio = np.frombuffer(acc, dtype=dtype)
        yield audio


# Taken from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


# Taken from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
def frame_generator(frame_duration_ms, audio_generator, sample_rate):
    """
    Generates audio frames from PCM audio data. Takes the desired frame duration in milliseconds, the PCM data, and the
    sample rate. Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    duration = (float(n) / sample_rate) / 2.0
    timestamp = 0.0
    for audio in audio_generator:
        offset = 0
        while offset + n < len(audio):
            yield Frame(audio[offset : offset + n], timestamp, duration)
            timestamp += duration
            offset += n


# Taken from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """
    Filters out non-voiced audio frames. Given a webrtcvad.Vad and a source of audio frames, yields only the voiced
    audio. Uses a padded, sliding window algorithm over the audio frames. When more than 90% of the frames in the
    window are voiced (as reported by the VAD), the collector triggers and begins yielding audio frames. Then the
    collector waits until 90% of the frames in the window are unvoiced to detrigger. The window is padded at the front
    and back to provide a small amount of silence or the beginnings/endings of speech around the voiced frames.

    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write("+(%s)" % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b"".join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b"".join([f.bytes for f in voiced_frames])


def vad_files(filenames, sampling_rate: int, chunk_length_s: float):
    try:
        import webrtcvad
    except ImportError:
        raise ValueError(
            "webrtcvad was not found but is required to chunk on voice activation, `pip install webrtcvad`."
        )

    for filename in filenames:
        if not isinstance(filename, str):
            raise ValueError("Chunk voice can only operate on large filenames")

        inputs = ffmpeg_stream(filename, sampling_rate, format_for_conversion="s16le", chunk_length_s=chunk_length_s)
        vad = webrtcvad.Vad(0)
        frames = frame_generator(10, inputs, sampling_rate)
        segments = vad_collector(sampling_rate, 10, 300, vad, frames)
        max_int16 = 2 ** 15
        max_len = int(round(chunk_length_s * sampling_rate))
        for i, segment in enumerate(segments):
            audio = np.frombuffer(segment, dtype=np.int16).astype("float32") / max_int16
            for i in range(0, audio.shape[0], max_len):
                chunk = audio[i : i + max_len]
                yield chunk


def chunk_files(
    filenames,
    sampling_rate: int,
    chunk_length_s: float,
    stride_length_s: Optional[Union[float, Tuple[float, float]]] = None,
):
    if stride_length_s is None:
        stride_length_s = [chunk_length_s / 6, chunk_length_s / 6]
    if isinstance(stride_length_s, (float, int)):
        stride_length_s = [stride_length_s, stride_length_s]

    chunk_len = int(round(sampling_rate * chunk_length_s))
    stride_left_ = int(round(sampling_rate * stride_length_s[0]))
    stride_right = int(round(sampling_rate * stride_length_s[1]))
    stride_left = 0

    for filename in filenames:
        if not isinstance(filename, str):
            raise ValueError("Chunk voice can only operate on large filenames")

        chunk = np.zeros((0,), dtype=np.float32)
        for audio in ffmpeg_stream(
            filename, sampling_rate, format_for_conversion="f32le", chunk_length_s=chunk_length_s
        ):
            chunk = np.concatenate([chunk, audio])
            while chunk.shape[0] >= chunk_len:
                to_send = chunk[:chunk_len]
                yield {"raw": to_send, "stride": (stride_left, stride_right)}
                # Start striding left
                if stride_left == 0:
                    stride_left = stride_left_
                chunk = chunk[chunk_len - stride_right - stride_left :]
        # last chunk only if there's actual content
        if chunk.shape[0] > stride_left:
            yield {"raw": chunk, "stride": (stride_left, 0)}
