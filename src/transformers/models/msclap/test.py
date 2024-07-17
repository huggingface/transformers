
import torch 
import random 
import numpy as np 

import torch
torch.manual_seed(0)
from msclap import CLAP
import torchaudio
import torchaudio.transforms as T

from torchlibrosa.stft import Spectrogram, LogmelFilterBank


audio_path = ['/home/kamil/transformers/runs/sample-0.mp3']

model_clap = CLAP(version = '2023', use_cuda=False)


sampling_rate= 44100
duration: 7
fmin =  50
fmax = 14000 #14000 
n_fft = 1024 # 1028 
hop_size =  320
mel_bins = 64
window_size =  1024
window = 'hann'
center = True
pad_mode = 'reflect'
ref = 1.0
amin = 1e-10
top_db = None

audio_time_series, sample_rate = torchaudio.load(audio_path[0])
        
resample_rate = 44100

if resample_rate != sample_rate:
    resampler = T.Resample(sample_rate, resample_rate)
    audio_time_series = resampler(audio_time_series)

audio_time_series = audio_time_series.reshape(-1)
sample_rate = resample_rate
# audio_time_series is shorter than predefined audio duration,
# so audio_time_series is extended
audio_duration = 7
if audio_duration*sample_rate >= audio_time_series.shape[0]:
    repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                audio_time_series.shape[0]))
    # Repeat audio_time_series by repeat_factor to match audio_duration
    audio_time_series = audio_time_series.repeat(repeat_factor)
    # remove excess part of audio_time_series
    audio_time_series = audio_time_series[0:audio_duration*sample_rate]
else:
    # audio_time_series is longer than predefined audio duration,
    # so audio_time_series is trimmed
    start_index = random.randrange(
        audio_time_series.shape[0] - audio_duration*sample_rate)
    audio_time_series = audio_time_series[start_index:start_index +
                                            audio_duration*sample_rate]

audio_tensor = audio_time_series 
audio_tensors = []
audio_tensor = audio_tensor.reshape(
                1, -1).cuda() if torch.cuda.is_available() else audio_tensor.reshape(1, -1)
audio_tensor =audio_tensor


spec = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True).to("cuda")

logmel_extractor = LogmelFilterBank(sr=32000, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True).to('cuda')

x1 = spec(audio_tensor)
x1 = logmel_extractor(x1)

x = model_clap.clap.audio_encoder.base.htsat.spectrogram_extractor(audio_tensor)
x = model_clap.clap.audio_encoder.base.htsat.logmel_extractor(x)

print(x == x1)

