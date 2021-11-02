import collections
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


def ffmpeg_stream(
    filename: str, sampling_rate: int, format_for_conversion: str = "f32le", chunk_max_duration_s: int = 10
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
        size_of_sample = 3
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

    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=bufsize)
    except FileNotFoundError:
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename")

    buflen = int(sampling_rate * chunk_max_duration_s * size_of_sample)
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

        # sys.stdout.write("1" if is_speech else "0")
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
                # sys.stdout.write("-(%s)" % (frame.timestamp + frame.duration))
                triggered = False
                yield b"".join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    # if triggered:
    #     sys.stdout.write("-(%s)" % (frame.timestamp + frame.duration))
    # sys.stdout.write("\n")
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b"".join([f.bytes for f in voiced_frames])
