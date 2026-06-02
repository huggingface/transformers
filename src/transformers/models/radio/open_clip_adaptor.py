# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from argparse import Namespace

from .adaptor_generic import GenericAdaptor
from .adaptor_registry import adaptor_registry, dict_t, state_t
from .utils import rank_gate


class OpenCLIP_RADIO(GenericAdaptor):
    def __init__(self, main_config: Namespace, adaptor_config: dict_t, state: state_t):
        super().__init__(main_config, adaptor_config, state)

        import open_clip

        with rank_gate():
            self.oc_model = open_clip.create_model_from_pretrained(
                model_name=adaptor_config["model"],
                pretrained=adaptor_config["pretrained"],
                return_transform=False,
            )
        # Unload these parameters
        self.oc_model.visual = None

        self.tokenizer = open_clip.get_tokenizer(model_name=adaptor_config["model"])

    def encode_text(self, text, normalize: bool = False):
        return self.oc_model.encode_text(text, normalize=normalize)


@adaptor_registry.register_adaptor("open_clip")
def create_open_clip_adaptor(main_config: Namespace, adaptor_config: dict_t, state: state_t):
    return OpenCLIP_RADIO(main_config, adaptor_config, state)
