# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch LLaMA model. """


import unittest
from typing import Union, Optional
from parameterized import parameterized
import inspect

from transformers import LlamaConfig, is_torch_available, set_seed,LlavaConfig
from transformers.models.llava.configuration_llava import LlavaLlamaConfig


from transformers.testing_utils import require_torch, require_torch_gpu, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin, 
    ids_tensor, 
    random_attention_mask,
    floats_tensor,
    _config_zero_init
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        LlavaLlamaForCausalLM,
        LlamaForSequenceClassification,
        LlamaModel,
        LlamaTokenizer,
    )


class LlamaModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    
    def get_config(self):
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
             config,
             input_ids,
             token_type_ids,
             input_mask,
             sequence_labels,
             token_labels,
             choice_labels,
         ) = config_and_inputs
        
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, input_ids

class LlavaModelTester:
    def __init__(
        self,
        parent,
        d_model:int = 4096,
        emb_pdrop:int = 0,
        embedding_fraction:float = 1.0,
        expansion_ratio:int = 4,
        freeze_mm_mlp_adapter:bool = False,
        init_device:str = "cpu",
        learned_pos_emb:bool = True,
        logit_scale: Optional[Union[float, str]] = None,
        max_seq_len:int = 2048,
        mm_hidden_size:int = 1024,
        mm_use_im_start_end:bool = True,
        mm_vision_select_layer:int = -2,
        mm_vision_tower:str = "openai/clip-vit-large-patch14",
        model_type:str = "llava_mpt",
        n_heads:int = 32,
        n_layers:int = 32,
        no_bias:bool = True,
        norm_type:str = "low_precision_layernorm",
        resid_pdrop:int = 0,
        sep_image_conv_front:bool = False,
        torch_dtype:str = "float16",
        tune_mm_mlp_adapter:bool = False,
        use_cache:bool = True,
        use_mm_proj:bool = True,
        verbose:int = 0,
        vocab_size:int = 50282,
        **kwargs,
    ):

        self.freeze_mm_mlp_adapter = freeze_mm_mlp_adapter
        self.mm_hidden_size = mm_hidden_size
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_vision_select_layer = mm_vision_select_layer
        self.sep_image_conv_front = sep_image_conv_front
        self.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        self.use_mm_proj = use_mm_proj
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        #self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.image_size = 32
        self.batch_size=13
        self.num_channels=3

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return LlavaConfig(
          vocab_size = self.vocab_size,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        return config, pixel_values


class LlavaLlamaModelTester:
    def __init__(
        self,
        parent,
        llama_kwargs=None,
        llava_kwargs=None,
        
    ):
        self.is_training = True
        
        if llama_kwargs is None:
            llama_kwargs = {}
        if llava_kwargs is None:
            llava_kwargs = {}


        self.parent = parent
        self.llama_model_tester = LlamaModelTester(parent, **llama_kwargs)
        self.llava_model_tester = LlavaModelTester(parent, **llava_kwargs)

    def prepare_config_and_inputs(self):  
        (
             config,
             input_ids,
             token_type_ids,
             input_mask,
             sequence_labels,
             token_labels,
             choice_labels,
        ) = self.llama_model_tester.prepare_config_and_inputs() 
        _, pixel_values = self.llava_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, pixel_values

    def get_config(self):
        return LlavaLlamaConfig.from_llava_llama_configs(
            llama_config = self.llama_model_tester.get_config(),
            llava_config = self.llava_model_tester.get_config()
        )

    def create_and_check_model(                                                                   
         self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
     ):
         model = LlamaModel(config=config)
         model.to(torch_device)
         model.eval()
         result = model(input_ids, attention_mask=input_mask)
         result = model(input_ids)
         self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        pixel_values,
        
    ):
        model = LlavaLlamaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, pixel_values=pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    @unittest.skip(reason="Blip2Model does not have input/output embeddings")
    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        pixel_values,
        
    ):
        pass

    
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            pixel_values,
            
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "pixel_values": pixel_values}
        return config, inputs_dict



@require_torch
class LlavaLlamaModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_generative_model_classes = (LlavaLlamaForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "text-generation": LlavaLlamaForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    

    def setUp(self):
        self.model_tester = LlavaLlamaModelTester(self)
    
    
    #def test_model_from_pretrained(self):
    #    for model_name in BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST:
    #        model = Blip2ForConditionalGeneration.from_pretrained(model_name)
    #        self.assertIsNotNone(model)
    
    def test_forward_signature(self): 
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic       
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_get_text_features(self):
        vocab_size=99
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {
            "input_ids": torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(torch_device),
            "attention_mask": torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(torch_device),
            "decoder_input_ids": torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(torch_device),
        }
        model = LlavaLlamaForCausalLM(config).to(torch_device)
        model.eval()
        text_features = model.get_text_features(**inputs_dict)
        self.assertEqual(text_features[0].shape, (1, 10, vocab_size))

    def test_get_image_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        keys_to_pop = ["input_ids"]

        for key in keys_to_pop:
            inputs_dict.pop(key)

        model = LlavaLlamaForCausalLM(config).to(torch_device)
        model.eval()
        image_features = model.get_image_features(**inputs_dict)
        self.assertEqual(
            image_features[0].shape,
            (
                self.model_tester.llava_model_tester.batch_size,
                self.model_tester.llava_model_tester.seq_length,
                config.llava_config.hidden_size,
            ),
        )

    # override from common to deal with nested configurations (`vision_config`, `text_config` and `qformer_config`)
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for key in ["llama_config", "llava_config"]:
            setattr(configs_no_init, key, _config_zero_init(getattr(configs_no_init, key)))
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                         [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )        

    @unittest.skip("LLaMA buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass


@require_torch
class LlamaIntegrationTest(unittest.TestCase):
    @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
    @slow
    def test_model_7b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = LlavaLlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
        out = model(torch.tensor([input_ids]))
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-6.6550, -4.1227, -4.9859, -3.2406, 0.8262, -3.0033, 1.2964, -3.3699]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        # fmt: off
        EXPECTED_SLICE = torch.tensor([-12.8281, -7.4453, -0.4639, -8.0625, -7.2500, -8.0000, -6.4883, -7.7695, -7.8438, -7.0312, -6.2188, -7.1328, -1.8496, 1.9961, -8.6250, -6.7227, -12.8281, -6.9492, -7.0742, -7.7852, -7.5820, -7.9062, -6.9375, -7.9805, -8.3438, -8.1562, -8.0469, -7.6250, -7.7422, -7.3398,])
        # fmt: on
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

    @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
    @slow
    def test_model_13b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = LlavaLlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto")
        out = model(torch.tensor(input_ids))
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-2.0622, -1.2794, -1.1638, -0.9788, -1.4603, -1.0238, -1.7893, -1.4411]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        # fmt: off
        EXPECTED_SLICE = torch.tensor([-8.1406, -8.0547, 2.7461, -1.2344, -0.1448, -1.8262, -1.0020, -1.8154, -1.6895, -1.8516, -2.3574, -0.9277, 3.7598, 6.5742, -1.2998, -0.1177, -8.1406, -2.9688, -2.9199, -3.1699, -3.5254, -2.3555, -2.7988, -3.4141, -2.8262, -4.5195, -3.3379, -3.3164, -2.7832, -3.0273])
        # fmt: on
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)



