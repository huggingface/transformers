# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging

import torch

from .mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14

logger = logging.getLogger(__name__)

def on_load_checkpoint(model, checkpoint: dict) -> None:
        state_dict = checkpoint
        model_state_dict = model.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logger.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

            
def mae_model(name, pretrained_weights, image_size, vocab_size, max_2d_position_embeddings, **kwargs):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    mae_models = {
        'mae_vit_base_patch16': mae_vit_base_patch16,
        'mae_vit_large_patch16': mae_vit_large_patch16,
        'mae_vit_huge_patch14': mae_vit_huge_patch14,
    }
    
    if name not in mae_models:
        raise RuntimeError(f'{name} is not available')
    
    model = mae_models[name](image_size=image_size, vocab_size=vocab_size, max_2d_position_embeddings=max_2d_position_embeddings)
    
    try:
        weights = torch.load(pretrained_weights, map_location='cpu')

        on_load_checkpoint(model, weights['model'])
        model.load_state_dict(weights['model'], strict=False)
    except Exception as e:
        print('failed loaded mae')
    return model
