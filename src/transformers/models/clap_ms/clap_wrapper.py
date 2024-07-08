from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import random
import torchaudio
import collections
import re
import numpy as np
from transformers import AutoTokenizer, logging
from clap import CLAP
# from .mapper import get_clapcap
import math
import torchaudio.transforms as T
import os
import torch
import argparse
import yaml
import sys
from huggingface_hub.file_download import hf_hub_download
logging.set_verbosity_error()


# class AudioFeatureExtractor: 

#     def __init__(self,  
#         version, 
#         ) -> None:
    

#         self.config_as_str = (Path(__file__).parent / f"configs/config_{version}.yml").read_text()
#         config = self.read_config_as_args(self.config_as_str, is_config_str=True)
#         config.sample_rate = config.sampling_rate
#         window = 'hann'
#         center = True
#         pad_mode = 'reflect'
#         ref = 1.0
#         amin = 1e-10
#         top_db = None
#         self.interpolate_ratio = 32 

#         # Spectrogram extractor
#         self.spectrogram_extractor = Spectrogram(n_fft=config.window_size, hop_length=config.hop_size, 
#             win_length=config.window_size, window=window, center=center, pad_mode=pad_mode, 
#             freeze_parameters=True)
#         # Logmel feature extractor
#         self.logmel_extractor = LogmelFilterBank(sr=config.sample_rate, n_fft=config.window_size, 
#             n_mels=config.mel_bins, fmin=config.fmin, fmax=config.fmax, ref=ref, amin=amin, top_db=top_db, 
#             freeze_parameters=True)
#         # Spec augmenter
#         self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
#             freq_drop_width=8, freq_stripes_num=2) # 2 2
        
#     def forward(self, x): 

#         x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
#         x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
#         return x 

#     def read_config_as_args(self,config_path,args=None,is_config_str=False):
#         return_dict = {}

#         if config_path is not None:
#             if is_config_str:
#                 yml_config = yaml.load(config_path, Loader=yaml.FullLoader)
#             else:
#                 with open(config_path, "r") as f:
#                     yml_config = yaml.load(f, Loader=yaml.FullLoader)

#             if args != None:
#                 for k, v in yml_config.items():
#                     if k in args.__dict__:
#                         args.__dict__[k] = v
#                     else:
#                         sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))
#             else:
#                 for k, v in yml_config.items():
#                     return_dict[k] = v

#         args = args if args != None else return_dict
#         return argparse.Namespace(**args)

class CLAPWrapper():
    """
    A class for interfacing CLAP model.  
    """
    model_repo = "microsoft/msclap"
    model_name = {
        '2022': 'CLAP_weights_2022.pth',
        '2023': 'CLAP_weights_2023.pth',
        'clapcap': 'clapcap_weights_2023.pth'
    }

    def __init__(self, model_fp: Path | str | None = None, version: str = '2023', use_cuda=False):
        # Check if version is supported
        self.supported_versions = self.model_name.keys()
        if version not in self.supported_versions:
            raise ValueError(f"The version {version} is not supported. The supported versions are {str(self.supported_versions)}")

        self.np_str_obj_array_pattern = re.compile(r'[SaUO]')
        self.file_path = os.path.realpath(__file__)
        self.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")
        self.config_as_str = (Path(__file__).parent / f"config_{version}.yml").read_text()

        # Automatically download model if not provided
        if not model_fp:
            model_fp = hf_hub_download(self.model_repo, self.model_name[version])
            
        self.model_fp = model_fp
        self.use_cuda = use_cuda
        if 'clapcap' in  version:
            self.clapcap, self.tokenizer, self.args = self.load_clapcap()
        else:
            self.clap, self.tokenizer, self.args = self.load_clap()
    
    def read_config_as_args(self,config_path,args=None,is_config_str=False):
        return_dict = {}

        if config_path is not None:
            if is_config_str:
                yml_config = yaml.load(config_path, Loader=yaml.FullLoader)
            else:
                with open(config_path, "r") as f:
                    yml_config = yaml.load(f, Loader=yaml.FullLoader)

            if args != None:
                for k, v in yml_config.items():
                    if k in args.__dict__:
                        args.__dict__[k] = v
                    else:
                        sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))
            else:
                for k, v in yml_config.items():
                    return_dict[k] = v

        args = args if args != None else return_dict
        return argparse.Namespace(**args)

    def load_clap(self):
        r"""Load CLAP model with args from config file"""

        args = self.read_config_as_args(self.config_as_str, is_config_str=True)

        if 'roberta' in args.text_model or 'clip' in args.text_model or 'gpt' in args.text_model:
            self.token_keys = ['input_ids', 'attention_mask']
        elif 'bert' in args.text_model:
            self.token_keys = ['input_ids', 'token_type_ids', 'attention_mask']

        clap = CLAP(
            audioenc_name=args.audioenc_name,
            sample_rate=args.sampling_rate,
            window_size=args.window_size,
            hop_size=args.hop_size,
            mel_bins=args.mel_bins,
            fmin=args.fmin,
            fmax=args.fmax,
            classes_num=args.num_classes,
            out_emb=args.out_emb,
            text_model=args.text_model,
            transformer_embed_dim=args.transformer_embed_dim,
            d_proj=args.d_proj
        )

        # Load pretrained weights for model
        model_state_dict = torch.load(self.model_fp, map_location=torch.device('cpu'))['model']

        # We unwrap the DDP model and save. If the model is not unwrapped and saved, then the model needs to unwrapped before `load_state_dict`: 
        # Reference link: https://discuss.pytorch.org/t/how-to-load-dataparallel-model-which-trained-using-multiple-gpus/146005
        clap.load_state_dict(model_state_dict, strict=False)

        clap.eval()  # set clap in eval mode
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        if 'gpt' in args.text_model:
            tokenizer.add_special_tokens({'pad_token': '!'})

        if self.use_cuda and torch.cuda.is_available():
            clap = clap.cuda()

        return clap, tokenizer, args
    
    # def load_clapcap(self):
    #     r"""Load CLAP model with args from config file"""

    #     args = self.read_config_as_args(self.config_as_str, is_config_str=True)
    #     args.prefix_dim = args.d_proj
    #     text_model = args.text_model
    #     args.text_model = args.text_decoder
    #     args.cross_attention = True if 'cross' in args.clapcap_model.lower() else False

    #     if 'roberta' in args.text_model or 'clip' in args.text_model or 'gpt' in args.text_model:
    #         self.token_keys = ['input_ids', 'attention_mask']
    #     elif 'bert' in args.text_model:
    #         self.token_keys = ['input_ids', 'token_type_ids', 'attention_mask']

    #     clap = CLAP(
    #         audioenc_name=args.audioenc_name,
    #         sample_rate=args.sampling_rate,
    #         window_size=args.window_size,
    #         hop_size=args.hop_size,
    #         mel_bins=args.mel_bins,
    #         fmin=args.fmin,
    #         fmax=args.fmax,
    #         classes_num=args.num_classes,
    #         out_emb=args.out_emb,
    #         text_model=text_model,
    #         transformer_embed_dim=args.transformer_embed_dim,
    #         d_proj=args.d_proj
    #     )

    #     clapcap = get_clapcap(args.clapcap_model)(clap, args.text_decoder, args.prefix_length, args.prefix_length_clip, args.prefix_dim,
    #              args.num_layers, args.normalize_prefix, args.mapping_type, True, True)

    #     model_state_dict = torch.load(self.model_fp, map_location=torch.device('cpu'))['model']
    #     clapcap.load_state_dict(model_state_dict, strict=False)

    #     clapcap.eval()  # set clap in eval mode
    #     tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    #     if 'gpt' in args.text_model:
    #         tokenizer.add_special_tokens({'pad_token': '!'})

    #     if self.use_cuda and torch.cuda.is_available():
    #         clapcap = clapcap.cuda()

    #     return clapcap, tokenizer, args

    def default_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))
    
    def read_audio(self, audio_path, resample=True):
        r"""Loads audio file or array and returns a torch tensor"""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        
        resample_rate = self.args.sampling_rate
        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        return audio_time_series, resample_rate

    def load_audio_into_tensor(self, audio_path, audio_duration, resample=False):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = self.read_audio(audio_path, resample=resample)
        audio_time_series = audio_time_series.reshape(-1)

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
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
        return torch.FloatTensor(audio_time_series)

    def preprocess_audio(self, audio_files, resample):
        r"""Load list of audio files and return raw audio"""
        audio_tensors = []
        for audio_file in audio_files:
            audio_tensor = self.load_audio_into_tensor(
                audio_file, self.args.duration, resample)
            audio_tensor = audio_tensor.reshape(
                1, -1).cuda() if self.use_cuda and torch.cuda.is_available() else audio_tensor.reshape(1, -1)
            audio_tensors.append(audio_tensor)
        return self.default_collate(audio_tensors)

    def preprocess_text(self, text_queries):
        r"""Load list of class labels and return tokenized text"""
        tokenized_texts = []
        for ttext in text_queries:
            if 'gpt' in self.args.text_model:
                ttext = ttext + ' <|endoftext|>'
            tok = self.tokenizer.encode_plus(
                text=ttext, add_special_tokens=True, max_length=self.args.text_len, padding='max_length', return_tensors="pt")
            for key in self.token_keys:
                tok[key] = tok[key].reshape(-1).cuda() if self.use_cuda and torch.cuda.is_available() else tok[key].reshape(-1)
            tokenized_texts.append(tok)
        return self.default_collate(tokenized_texts)

    def get_text_embeddings(self, class_labels):
        r"""Load list of class labels and return text embeddings"""
        preprocessed_text = self.preprocess_text(class_labels)
        return self._get_text_embeddings(preprocessed_text)

    def get_audio_embeddings(self, audio_files, resample=True):
        r"""Load list of audio files and return a audio embeddings"""
        preprocessed_audio = self.preprocess_audio(audio_files, resample)
        return self._get_audio_embeddings(preprocessed_audio)

    def _get_text_embeddings(self, preprocessed_text):
        r"""Load preprocessed text and return text embeddings"""
        with torch.no_grad():
            return self.clap.caption_encoder(preprocessed_text)

    def _get_audio_embeddings(self, preprocessed_audio):
        r"""Load preprocessed audio and return a audio embeddings"""
        with torch.no_grad():
            preprocessed_audio = preprocessed_audio.reshape(
                preprocessed_audio.shape[0], preprocessed_audio.shape[2])
            #Append [0] the audio emebdding, [1] has output class probabilities
            return self.clap.audio_encoder(preprocessed_audio)[0]

    def _generic_batch_inference(self, func, *args):
        r"""Process audio and/or text per batch"""
        input_tmp = args[0]
        batch_size = args[-1]
        # args[0] has audio_files, args[1] has class_labels
        inputs = [args[0], args[1]] if len(args) == 3 else [args[0]]
        args0_len = len(args[0])
        # compute text_embeddings once for all the audio_files batches
        if len(inputs) == 2:
            text_embeddings = self.get_text_embeddings(args[1])
            inputs = [args[0], args[1], text_embeddings]
        dataset_idx = 0
        for _ in range(math.ceil(args0_len/batch_size)):
            next_batch_idx = dataset_idx + batch_size
            # batch size is bigger than available audio/text items
            if next_batch_idx >= args0_len:
                inputs[0] = input_tmp[dataset_idx:]
                yield func(*tuple(inputs))
            else:
                inputs[0] = input_tmp[dataset_idx:next_batch_idx]
                yield func(*tuple(inputs))
            dataset_idx = next_batch_idx

    def get_audio_embeddings_per_batch(self, audio_files, batch_size):
        r"""Load preprocessed audio and return a audio embeddings per batch"""
        return self._generic_batch_inference(self.get_audio_embeddings, audio_files, batch_size)

    def get_text_embeddings_per_batch(self, class_labels, batch_size):
        r"""Load preprocessed text and return text embeddings per batch"""
        return self._generic_batch_inference(self.get_text_embeddings, class_labels, batch_size)

    def compute_similarity(self, audio_embeddings, text_embeddings):
        r"""Compute similarity between text and audio embeddings"""
        audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
        text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
    
        logit_scale = self.clap.logit_scale.exp()
        similarity = logit_scale*text_embeddings @ audio_embeddings.T
        return similarity.T

    def classify_audio_files_per_batch(self, audio_files, class_labels, batch_size):
        r"""Compute classification probabilities for each audio recording in a batch and each class label"""
        return self._generic_batch_inference(self.classify_audio_files, audio_files, class_labels, batch_size)
    
    def generate_caption(self, audio_files, resample=True, beam_size: int = 5, entry_length=67, temperature=1.):
        r"""Generate audio captions for each audio recording in a batch"""
        captions = []
        audio_tensors = self.preprocess_audio(audio_files, resample)

        with torch.no_grad():
            prefix = self.clapcap.clap(audio_tensors.squeeze(1))[0]
            if self.args.normalize_prefix:
                prefix = prefix / prefix.norm(2, -1).reshape(-1,1)
            prefix_embed = self.clapcap.clap_project(prefix).view(-1, self.args.prefix_length, self.clapcap.gpt.transformer.wte.weight.shape[1])

            for i in range(len(audio_tensors)):
                gen_caption = self._generate_beam(embed=prefix_embed[i].unsqueeze(0),\
                                                            beam_size=beam_size,\
                                                            entry_length=entry_length,\
                                                            temperature=temperature)[0]
                captions.append(gen_caption.capitalize())
        return captions
    
    def _generate_beam(self, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = ' <|endoftext|>'):
        r"""Generate captions by beam search decoding"""
        self.clapcap.eval()
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        tokens = None
        scores = None
        device = next(self.clapcap.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(self.tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                    generated = self.clapcap.gpt.transformer.wte(tokens)
            for i in range(entry_length):
                outputs = self.clapcap.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = self.clapcap.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [self.tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts
