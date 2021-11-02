import collections
import platform
import subprocess

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


def ffmpeg_stream(filename: str, sampling_rate: int, format_for_conversion: str, chunk_max_duration_s: int):
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

    buflen = int(sampling_rate * chunk_max_duration_s * size_of_sample)
    return _ffmpeg_stream(ffmpeg_command, bufsize, buflen, dtype)


def ffmpeg_microphone(sampling_rate: int, format_for_conversion: str, chunk_max_duration_s: int):
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
        format_ = "pulse"
    elif system == "Darwin":
        format_ = "avfoundation"
    elif system == "Windows":
        format_ = "dshow"

    ffmpeg_command = [
        "ffmpeg",
        "-f",
        format_,
        "-i",
        "default",
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

    buflen = int(sampling_rate * chunk_max_duration_s * size_of_sample)
    return _ffmpeg_stream(ffmpeg_command, bufsize, buflen, dtype)


def _ffmpeg_stream(ffmpeg_command, bufsize, buflen, dtype):
    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=bufsize)
    except FileNotFoundError:
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename")

    running = True
    while running:
        raw = ffmpeg_process.stdout.read(buflen)
        if raw == b"":
            running = False
            break

        audio = np.frombuffer(raw, dtype=dtype)
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


def vad_files(filenames, sampling_rate: int, max_chunk_duration_s: int):
    try:
        import webrtcvad
    except ImportError:
        raise ValueError(
            "webrtcvad was not found but is required to chunk on voice activation, `pip install webrtcvad`."
        )

    for filename in filenames:
        if not isinstance(filename, str):
            raise ValueError("Chunk voice can only operate on large filenames")

        inputs = ffmpeg_stream(
            filename, sampling_rate, format_for_conversion="s16le", chunk_max_duration_s=max_chunk_duration_s
        )
        vad = webrtcvad.Vad(0)
        frames = frame_generator(10, inputs, sampling_rate)
        segments = vad_collector(sampling_rate, 10, 300, vad, frames)
        max_int16 = 2 ** 15
        max_len = int(max_chunk_duration_s * sampling_rate)
        for i, segment in enumerate(segments):
            audio = np.frombuffer(segment, dtype=np.int16).astype("float32") / max_int16
            for i in range(0, audio.shape[0], max_len):
                chunk = audio[i : i + max_len]
                yield chunk


def chunk_files(filenames, sampling_rate: int, max_chunk_duration_s: int):
    try:
        from scipy import signal
    except ImportError:
        raise ValueError("scipy was not found but is required to chunk on voice activation, `pip install scipy`.")

    f1 = 50  # 50Hz
    f2 = 300  # 300Hz
    fs = sampling_rate

    nyq = 0.5 * fs
    low = f1 / nyq
    high = f2 / nyq
    order = 10
    sos = signal.butter(order, [low, high], analog=False, btype="band", output="sos")
    chunk_min_duration = 5
    chunk_pad_duration = 0.3
    start_chunk = int(sampling_rate * chunk_min_duration)
    pad_chunk = int(sampling_rate * chunk_pad_duration)

    for filename in filenames:
        leftover = np.zeros((0,), dtype=np.float32)
        pad = np.zeros((pad_chunk,), dtype=np.float32)
        if not isinstance(filename, str):
            raise ValueError("Chunk voice can only operate on large filenames")

        for audio in ffmpeg_stream(
            filename, sampling_rate, format_for_conversion="f32le", max_chunk_duration_s=max_chunk_duration_s
        ):
            audio = np.concatenate([leftover, audio])
            chunk_portion = audio[start_chunk:]
            if chunk_portion.shape[0] == 0:
                padded = np.concatenate([pad, audio, pad])
                yield padded
                break
            voice_filtered = signal.sosfilt(sos, chunk_portion)
            index = start_chunk + voice_filtered.argmin()
            chunked = audio[:index]

            leftover = audio[index:]

            padded = np.concatenate([pad, chunked, pad])
            yield padded
