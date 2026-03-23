# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""
NEFTune: Noisy Embeddings for Fine-Tuning.

Implementation based on https://github.com/neelsjain/NEFTune
Paper: https://huggingface.co/papers/2310.05914
"""

import torch

from ..trainer_utils import _is_peft_model


def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for torch.nn.Embedding
    layers. This method is slightly adapted from the original source code that can be found here:
    https://github.com/neelsjain/NEFTune. Simply add it to your model as follows:
    ```python
    from transformers.integrations.neftune import neftune_post_forward_hook

    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```
    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set `module.neftune_noise_alpha` to
            the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    return output


def activate_neftune(model, neftune_noise_alpha, accelerator=None):
    """
    Activates NEFTune (Noisy Embeddings for Fine-Tuning) on the model.

    NEFTune adds noise to embedding vectors during training, which has been shown to improve
    fine-tuning performance. See https://huggingface.co/papers/2310.05914 for details.

    Args:
        model (`torch.nn.Module`):
            The model to activate NEFTune on.
        neftune_noise_alpha (`float`):
            The noise alpha value controlling the magnitude of the noise.
        accelerator (`Accelerator`, *optional*):
            The accelerator instance. If provided, the model will be unwrapped before
            accessing embeddings. Required when using distributed training.

    Returns:
        `torch.utils.hooks.RemovableHandle`: The hook handle that can be used to deactivate NEFTune.
    """
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
    else:
        unwrapped_model = model

    if _is_peft_model(unwrapped_model):
        embeddings = unwrapped_model.base_model.model.get_input_embeddings()
    else:
        embeddings = unwrapped_model.get_input_embeddings()

    embeddings.neftune_noise_alpha = neftune_noise_alpha
    hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)

    return hook_handle


def deactivate_neftune(model, hook_handle, accelerator=None):
    """
    Deactivates NEFTune on the model.

    Args:
        model (`torch.nn.Module`):
            The model to deactivate NEFTune on.
        hook_handle (`torch.utils.hooks.RemovableHandle`):
            The hook handle returned by `activate_neftune`.
        accelerator (`Accelerator`, *optional*):
            The accelerator instance. If provided, the model will be unwrapped before
            accessing embeddings.
    """
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
    else:
        unwrapped_model = model

    if _is_peft_model(unwrapped_model):
        embeddings = unwrapped_model.base_model.model.get_input_embeddings()
    else:
        embeddings = unwrapped_model.get_input_embeddings()

    hook_handle.remove()
    if hasattr(embeddings, "neftune_noise_alpha"):
        del embeddings.neftune_noise_alpha
