# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
""" Classes to support Encoder-Decoder architectures """


def prepare_encoder_decoder_model_kwargs(**kwargs):
    """ Prepare the encoder and decoder's keyword arguments.

    Keyword arguments come in 3 flavors:
    - encoder-specific (prefixed by `encoder_`)
    - decoder-specific (prefixed by `decoder_`)
    - those that apply to the model as whole.

    We let the specific kwargs override the common ones in case of
    conflict.
    """

    kwargs_common = {
        argument: value
        for argument, value in kwargs.items()
        if not argument.startswith("encoder_") and not argument.startswith("decoder_")
    }
    if "input_ids" in kwargs_common:
        kwargs["encoder_input_ids"] = kwargs_common.pop("input_ids")

    decoder_kwargs = kwargs_common.copy()
    encoder_kwargs = kwargs_common.copy()
    encoder_kwargs.update(
        {argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")}
    )
    decoder_kwargs.update(
        {argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")}
    )
    decoder_kwargs["encoder_attention_mask"] = encoder_kwargs.get("attention_mask", None)
    return encoder_kwargs, decoder_kwargs
