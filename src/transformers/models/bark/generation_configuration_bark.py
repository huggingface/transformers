# coding=utf-8
# Copyright 2023 The Suno AI Authors and The HuggingFace Inc. team. All rights reserved.
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
""" BARK model generation configuration"""

import copy
from typing import Dict

from ...generation.configuration_utils import GenerationConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class BarkSemanticGenerationConfig(GenerationConfig):
    model_type = "semantic"

    def __init__(
        self,
        eos_token_id=10_000,
        renormalize_logits=True,
        max_new_tokens=768,
        output_scores=False,
        return_dict_in_generate=False,
        output_hidden_states=False,
        output_attentions=False,
        temperature=0.7,
        do_sample=True,
        text_encoding_offset=10_048,
        text_pad_token=129_595,
        semantic_infer_token=129_599,
        semantic_vocab_size=10_000,
        max_input_semantic_length=256,
        semantic_rate_hz=49.9,
        **kwargs,
    ):
        """Class that holds a generation configuration for [`BarkSemanticModel`].

        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the
        documentation from [`GenerationConfig`] for more information.

        Args:
            eos_token_id (`int`, *optional*, defaults to 10_000): Same function as [`GenerationConfig`].
            renormalize_logits (`bool`, *optional*, defaults to True): Same function as [`GenerationConfig`].
            max_new_tokens (`int`, *optional*, defaults to 768): Same function as [`GenerationConfig`].
            output_scores (`bool`, *optional*, defaults to False): Same function as [`GenerationConfig`].
            return_dict_in_generate (`bool`, *optional*, defaults to False): Same function as [`GenerationConfig`].
            output_hidden_states (`bool`, *optional*, defaults to False): Same function as [`GenerationConfig`].
            output_attentions (`bool`, *optional*, defaults to False): Same function as [`GenerationConfig`].
            temperature (`float`, *optional*, defaults to 0.7): Same function as [`GenerationConfig`].
            do_sample (`bool`, *optional*, defaults to True): Same function as [`GenerationConfig`].
            text_encoding_offset (`int`, *optional*, defaults to 10_048): Text encoding offset.
            text_pad_token (`int`, *optional*, defaults to 129_595): Text pad token.
            semantic_infer_token (`int`, *optional*, defaults to 129_599): Semantic infer token.
            semantic_vocab_size (`int`, *optional*, defaults to 10_000): Semantic vocab size.
            max_input_semantic_length (`int`, *optional*, defaults to 256): Max lenght of semantic input vector.
            semantic_rate_hz (`float`, *optional*, defaults to 49.9): Semantic rate in Hertz.
        """
        super().__init__(
            temperature=temperature,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            renormalize_logits=renormalize_logits,
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )

        self.text_encoding_offset = text_encoding_offset
        self.text_pad_token = text_pad_token
        self.semantic_pad_token = eos_token_id
        self.semantic_infer_token = semantic_infer_token
        self.semantic_vocab_size = semantic_vocab_size
        self.max_input_semantic_length = max_input_semantic_length
        self.semantic_rate_hz = semantic_rate_hz

        # 256 et 257 + 1->
        # eos_token_id=self.config.semantic_pad_token,


class BarkCoarseGenerationConfig(GenerationConfig):
    model_type = "coarse_acoustics"

    def __init__(
        self,
        renormalize_logits=True,
        output_scores=False,
        return_dict_in_generate=False,
        output_hidden_states=False,
        output_attentions=False,
        temperature=0.7,
        do_sample=True,
        coarse_semantic_pad_token=12_048,
        coarse_rate_hz=75,
        n_coarse_codebooks=2,
        coarse_infer_token=12_050,
        max_coarse_input_length=256,
        max_coarse_history: int = 630,
        sliding_window_len: int = 60,
        **kwargs,
    ):
        """_summary_

        Args:
            renormalize_logits (`bool`, *optional*, defaults to True): Same function as [`GenerationConfig`].
            output_scores (`bool`, *optional*, defaults to False): Same function as [`GenerationConfig`].
            return_dict_in_generate (`bool`, *optional*, defaults to False): Same function as [`GenerationConfig`].
            output_hidden_states (`bool`, *optional*, defaults to False): Same function as [`GenerationConfig`].
            output_attentions (`bool`, *optional*, defaults to False): Same function as [`GenerationConfig`].
            temperature (`float`, *optional*, defaults to 0.): Same function as [`GenerationConfig`].
            do_sample (`bool`, *optional*, defaults to True): Same function as [`GenerationConfig`].
            coarse_semantic_pad_token (`int`, *optional*, defaults to 12_048): Coarse semantic pad token.
            coarse_rate_hz (`int`, *optional*, defaults to 75): Coarse rate in Hertz.
            n_coarse_codebooks (`int`, *optional*, defaults to 2): Number of coarse codebooks.
            coarse_infer_token (`int`, *optional*, defaults to 12_050): Coarse infer token.
            max_coarse_input_length (`int`, *optional*, defaults to 256): Max length of input coarse vector.
            max_coarse_history (`int`, *optional*, defaults to 630):
                Max length of the output of the coarse acoustics model used in the fine generation step.
            sliding_window_len (`int`, *optional*, defaults to 60):
                The coarse generation step uses a sliding window to generate raw audio.
        """
        super().__init__(
            temperature=temperature,
            do_sample=do_sample,
            renormalize_logits=renormalize_logits,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )

        self.coarse_semantic_pad_token = coarse_semantic_pad_token
        self.coarse_rate_hz = coarse_rate_hz
        self.n_coarse_codebooks = n_coarse_codebooks
        self.coarse_infer_token = coarse_infer_token
        self.max_coarse_input_length = max_coarse_input_length
        self.max_coarse_history = max_coarse_history
        self.sliding_window_len = sliding_window_len

        # 256


class BarkFineGenerationConfig(GenerationConfig):
    model_type = "fine_acoustics"

    def __init__(
        self,
        temperature=0.5,
        max_fine_history_length=512,
        max_fine_input_length=1024,
        n_fine_codebooks=8,
        **kwargs,
    ):
        """_summary_

        Args:
            temperature (`float`, *optional*, defaults to 0.5): Same function as [`GenerationConfig`].
            max_fine_history_length (`int`, *optional*, defaults to 512): Max length of the fine history vector.
            max_fine_input_length (`int`, *optional*, defaults to 1024): Max length of fine input vector.
            n_fine_codebooks (`int`, *optional*, defaults to 8): Number of codebooks used.
        """
        super().__init__(temperature=temperature)

        self.max_fine_history_length = max_fine_history_length
        self.max_fine_input_length = max_fine_input_length
        self.n_fine_codebooks = n_fine_codebooks


class BarkGenerationConfig(GenerationConfig):
    model_type = "bark"
    is_composition = True

    # TODO (joao): nested from_dict

    def __init__(
        self,
        semantic_config: Dict = None,
        coarse_acoustics_config: Dict = None,
        fine_acoustics_config: Dict = None,
        sample_rate=24_000,
        codebook_size=1024,
        **kwargs,
    ):
        """_summary_

        Args:
            semantic_config (`Dict`, *optional*, defaults to None): Semantic generation configuration.
            coarse_acoustics_config (`Dict`, *optional*, defaults to None): Coarse generation configuration.
            fine_acoustics_config (`Dict`, *optional*, defaults to None): Fine generation configuration.
            sample_rate (`int`, *optional*, defaults to 24_000): Sample rate.
            codebook_size (`int`, *optional*, defaults to 1024): Vector length for each codebook.
        """
        if semantic_config is None:
            semantic_config = {}
            logger.info("semantic_config is None. initializing the semantic model with default values.")

        if coarse_acoustics_config is None:
            coarse_acoustics_config = {}
            logger.info("coarse_acoustics_config is None. initializing the coarse model with default values.")

        if fine_acoustics_config is None:
            fine_acoustics_config = {}
            logger.info("fine_acoustics_config is None. initializing the fine model with default values.")

        self.semantic_config = BarkSemanticGenerationConfig(**semantic_config)
        self.coarse_acoustics_config = BarkCoarseGenerationConfig(**coarse_acoustics_config)
        self.fine_acoustics_config = BarkFineGenerationConfig(**fine_acoustics_config)

        self.sample_rate = sample_rate
        self.codebook_size = codebook_size

    @classmethod
    def from_sub_model_configs(
        cls,
        semantic_config: BarkSemanticGenerationConfig,
        coarse_acoustics_config: BarkCoarseGenerationConfig,
        fine_acoustics_config: BarkFineGenerationConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`BarkGenerationConfig`] (or a derived class) from bark sub-models generation configuration.

        Returns:
            [`BarkGenerationConfig`]: An instance of a configuration object
        """
        return cls(
            semantic_config=semantic_config.to_dict(),
            coarse_acoustics_config=coarse_acoustics_config.to_dict(),
            fine_acoustics_config=fine_acoustics_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        output["semantic_config"] = self.semantic_config.to_dict()
        output["coarse_acoustics_config"] = self.coarse_acoustics_config.to_dict()
        output["fine_acoustics_config"] = self.fine_acoustics_config.to_dict()

        output["model_type"] = self.__class__.model_type
        return output
