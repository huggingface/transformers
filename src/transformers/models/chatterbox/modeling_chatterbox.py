# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Chatterbox model - Complete TTS Pipeline."""

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from ...modeling_utils import PreTrainedModel
from ...models.s3gen.modeling_s3gen import S3GenModel
from ...models.t3.modeling_t3 import T3Cond, T3Model
from .configuration_chatterbox import ChatterboxConfig


logger = logging.getLogger(__name__)


def drop_invalid_tokens(speech_tokens):
    """Remove invalid tokens from speech token sequence."""
    if isinstance(speech_tokens, torch.Tensor):
        # Remove start/stop tokens and any invalid tokens
        valid_mask = speech_tokens < 6561
        return speech_tokens[valid_mask]
    return speech_tokens


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or containing chars not seen often in the dataset.
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalize first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (
            """, '"'),
        (""",
            '"',
        ),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


class ChatterboxPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ChatterboxConfig
    base_model_prefix = "chatterbox"
    supports_gradient_checkpointing = False


@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict


class ChatterboxModel(ChatterboxPreTrainedModel):
    """
    Complete Chatterbox TTS Pipeline Model.

    This model combines T3, S3Gen, and HiFTNet to provide a complete text-to-speech pipeline:
    1. Text tokens → T3 → Speech tokens
    2. Speech tokens → S3Gen → Mel spectrogram
    3. Mel spectrogram → HiFTNet → Waveform
    """

    def __init__(self, config: ChatterboxConfig):
        super().__init__(config)
        self.config = config

        # Initialize sub-models
        logger.info("Initializing T3 model...")
        self.t3 = T3Model(config.t3_config)

        logger.info("Initializing S3Gen model...")
        self.s3gen = S3GenModel(config.s3gen_config)

        # Sampling rates
        self.s3_sr = 16000  # S3 tokenizer sampling rate
        self.s3gen_sr = 24000  # S3Gen output sampling rate

        # Text tokenizer
        self.text_tokenizer = None

        # Post init
        self.post_init()

    def load_text_tokenizer(self, tokenizer_path):
        """Load text tokenizer from tokenizer.json file."""
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            logger.warning(f"Tokenizer file not found: {tokenizer_path}")
            return False

        try:
            self.text_tokenizer = Tokenizer.from_file(str(tokenizer_path))
            logger.info(f"✓ Loaded text tokenizer from: {tokenizer_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return False

    @property
    def device(self):
        """Get device of the model."""
        return next(self.parameters()).device

    def prepare_text_tokens(self, text: str, tokenizer=None) -> torch.Tensor:
        """
        Prepare text tokens from raw text.

        Args:
            text: Input text string
            tokenizer: Text tokenizer (if None, uses self.text_tokenizer or dummy tokens)

        Returns:
            Text tokens with start/stop markers
        """
        text = punc_norm(text)

        # Use provided tokenizer, or self.text_tokenizer, or create dummy
        if tokenizer is not None:
            if hasattr(tokenizer, "encode"):
                # HuggingFace tokenizers-style
                encoding = tokenizer.encode(text)
                text_tokens = torch.tensor([encoding.ids], dtype=torch.long)
            else:
                text_tokens = tokenizer.text_to_tokens(text)
        elif self.text_tokenizer is not None:
            # Use loaded tokenizer
            encoding = self.text_tokenizer.encode(text)
            text_tokens = torch.tensor([encoding.ids], dtype=torch.long)
        else:
            # For testing: create dummy tokens
            logger.warning("No tokenizer provided, using dummy tokens")
            num_tokens = min(len(text.split()), 50)
            text_tokens = torch.randint(1, self.config.t3_config.text_tokens_dict_size - 1, (1, num_tokens))

        # Add start/stop tokens if not already present
        sot = self.config.t3_config.start_text_token
        eot = self.config.t3_config.stop_text_token

        # Check if start/stop tokens are already in the sequence
        has_start = (text_tokens[0, 0] == sot).item() if text_tokens.numel() > 0 else False
        has_stop = (text_tokens[0, -1] == eot).item() if text_tokens.numel() > 0 else False

        if not has_start:
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        if not has_stop:
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        return text_tokens.to(self.device)

    def prepare_conditionals(
        self, reference_wav: np.ndarray, reference_sr: int, exaggeration: float = 0.5
    ) -> Conditionals:
        """
        Mirror the original Chatterbox prepare_conditionals method for parity.
        """
        if reference_wav.ndim == 1:
            ref_np = reference_wav
        else:
            ref_np = reference_wav.squeeze()

        # Prepare audio for S3Gen (24kHz) and T3 components (16kHz)
        if reference_sr != self.s3gen_sr:
            ref_24k = librosa.resample(ref_np, orig_sr=reference_sr, target_sr=self.s3gen_sr)
        else:
            ref_24k = ref_np

        if reference_sr != self.s3_sr:
            ref_16k = librosa.resample(ref_np, orig_sr=reference_sr, target_sr=self.s3_sr)
        else:
            ref_16k = ref_np

        # Truncate for conditioning lengths
        dec_len = 10 * self.s3gen_sr
        enc_len = 6 * self.s3_sr
        ref_24k = ref_24k[:dec_len]
        ref_16k = ref_16k[:enc_len]

        # Compute S3Gen conditioning dict
        ref_tensor_24k = torch.from_numpy(ref_24k).unsqueeze(0).to(self.device)
        with torch.no_grad():
            s3gen_ref_dict = self.s3gen.embed_ref(ref_tensor_24k, self.s3gen_sr, device=self.device)

        # Voice encoder speaker embedding
        ve_embed = self.t3.voice_encoder.embeds_from_wavs([ref_16k], sample_rate=self.s3_sr)
        speaker_emb = torch.from_numpy(ve_embed).to(self.device)

        # Speech prompt tokens for T3
        cond_prompt_speech_tokens = None
        if self.config.t3_config.speech_cond_prompt_len > 0:
            ref_tensor_16k = torch.from_numpy(ref_16k).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prompt_tokens, _ = self.s3gen.tokenizer(
                    ref_tensor_16k, return_dict=False, max_len=self.config.t3_config.speech_cond_prompt_len
                )
            cond_prompt_speech_tokens = prompt_tokens.to(self.device)

        # Build T3 conditioning
        emotion_adv = exaggeration * torch.ones(1, 1, 1, device=self.device)
        t3_cond = T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=cond_prompt_speech_tokens,
            emotion_adv=emotion_adv,
        )

        return Conditionals(t3=t3_cond, gen=s3gen_ref_dict)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        reference_wav: np.ndarray,
        reference_sr: int,
        tokenizer=None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        cfg_weight: float = 0.5,
        max_new_tokens: int = 1000,
        return_intermediates: bool = False,
    ):
        """
        Generate speech from text using the complete pipeline.

        Args:
            text: Input text to synthesize
            reference_wav: Reference audio for voice cloning (numpy array)
            reference_sr: Sampling rate of reference audio
            tokenizer: Optional text tokenizer
            exaggeration: Emotion/expressiveness level (0.0 to 1.0)
            temperature: Sampling temperature for T3
            top_p: Top-p sampling for T3
            min_p: Min-p sampling for T3
            repetition_penalty: Repetition penalty for T3
            cfg_weight: Classifier-free guidance weight for T3
            max_new_tokens: Maximum speech tokens to generate
            return_intermediates: Whether to return intermediate outputs (tokens, mel)

        Returns:
            Waveform tensor, or tuple of (waveform, intermediates) if return_intermediates=True
        """
        logger.info(f"Generating speech for text: '{text}'")

        # Step 1: Prepare text tokens
        text_tokens = self.prepare_text_tokens(text, tokenizer)

        # Step 2: Prepare conditionals (T3 + S3Gen) as in original pipeline
        conds = self.prepare_conditionals(reference_wav, reference_sr, exaggeration)
        t3_cond = conds.t3

        # For CFG: duplicate text tokens before passing to T3
        t3_text_tokens = text_tokens[0]  # Remove batch dimension
        if cfg_weight > 0.0:
            t3_text_tokens = torch.cat([t3_text_tokens.unsqueeze(0), t3_text_tokens.unsqueeze(0)], dim=0)

        speech_tokens = self.t3.inference(
            t3_cond=t3_cond,
            text_tokens=t3_text_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            cfg_weight=cfg_weight,
        )

        # Extract conditional batch (first sequence)
        if speech_tokens.dim() > 1:
            speech_tokens = speech_tokens[0]

        # Clean up speech tokens
        speech_tokens = drop_invalid_tokens(speech_tokens)

        # Additional safety check - ensure all tokens are valid
        if speech_tokens.max() >= 6561:
            speech_tokens = speech_tokens[speech_tokens < 6561]

        # Step 3: Generate waveform with S3Gen using prepared reference dict
        # Mirror original chatterbox: use s3gen.inference() which handles mel + vocoding
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict={k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in conds.gen.items()},
            finalize=True,
        )

        # Remove batch dimension
        wav = wav.squeeze(0)

        if return_intermediates:
            intermediates = {
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
            }
            return wav, intermediates

        return wav

    def forward(
        self,
        text: str,
        reference_wav: np.ndarray,
        reference_sr: int,
        tokenizer=None,
        **kwargs,
    ):
        """Forward pass - calls generate."""
        return self.generate(text, reference_wav, reference_sr, tokenizer, **kwargs)
