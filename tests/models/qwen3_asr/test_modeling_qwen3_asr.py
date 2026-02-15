import json
import unittest
import torch
import pytest
from pathlib import Path
from transformers import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    AutoProcessor,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    torch_device,
)
#from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


class Qwen3ASRModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 3
        self.seq_length = 10
        self.audio_token_id = 0

        self.text_config = {
            "model_type": "Qwen3ASRTextConfig",
            "vocab_size": 99,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "pad_token_id": 1,
        }

        self.audio_config = {
            "model_type": "Qwen3ASRAudioEncoderConfig",
            "d_model": 32,
            "encoder_layers": 2,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
        }

    def get_config(self):
        return Qwen3ASRConfig(
            thinker_config={
                "audio_config": self.audio_config,
                "text_config": self.text_config,
            },
            audio_token_id=self.audio_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.thinker_config.text_config.vocab_size)
        attention_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.long)
        #input_features = torch.randn(self.batch_size, num_mel_bins, feature_seq_len)
        #feature_attention_mask = torch.ones(self.batch_size, feature_seq_len, dtype=torch.long)
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            #"input_features": input_ids,
            #"feature_attention_mask": feature_attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()
        #config, input_features_values, input_features_mask = self.prepare_config_and_inputs()
        #num_audio_tokens_per_batch_idx = 8
        #input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        #attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        #attention_mask[:, :1] = 0
        #input_ids[:, 1 : 1 + num_audio_tokens_per_batch_idx] = config.audio_token_id
        #inputs_dict = {
        #    "input_ids": input_ids,
        #    "attention_mask": attention_mask,
        #    "input_features": input_features_values,
        #    "input_features_mask": input_features_mask,
        #}
        #input_dict = 0 #TODO
        #return config, inputs_dict


@require_torch
class Qwen3ASRForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):#GenerationTesterMixin, 
    all_model_classes = (Qwen3ASRForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {
        "automatic-speech-recognition": Qwen3ASRForConditionalGeneration,
    } if is_torch_available() else {}
    
    def setUp(self):
        self.model_tester = Qwen3ASRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3ASRConfig)

    @unittest.skip(
        reason="This test does not apply to Qwen3ASR since inputs_embeds corresponding to audio tokens are replaced when input features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in Qwen3ASR models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in Qwen3ASR models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="???")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass













    
@require_torch
class Qwen3ASRForConditionalGenerationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "Qwen/Qwen3-ASR-0.6B"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_integration(self):
        """
        This is an end-to-end integration test that verifies the model produces exactly the expected transcription 
        (both token IDs and decoded text) for a fixed audio input.
        """
        torch.manual_seed(0)
        path = Path(__file__).parent.parent.parent / "fixtures/qwen3_asr/expected_results.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        exp_ids = torch.tensor(raw["token_ids"])
        exp_txt = raw["transcriptions"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful ASR assistant."
                    },
                    {
                        "type": "audio",
                        "path": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
                    }
                ]
            }
        ]

        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            self.checkpoint, 
            device_map=torch_device, 
            dtype=torch.bfloat16
        ).eval()

        batch = self.processor.apply_chat_template(
            conversation, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=model.dtype)

        seq = model.generate(
            **batch, 
            max_new_tokens=64, 
            do_sample=False
        ).sequences

        inp_len = batch["input_ids"].shape[1]
        gen_ids = seq[:, inp_len:] if seq.shape[1] >= inp_len else seq

        txt = self.processor.batch_decode(
            seq, 
            skip_special_tokens=True
        )#[0].split("<asr_text>")[-1]

        torch.testing.assert_close(gen_ids.cpu(), exp_ids)  # 47 vs 263
        self.assertListEqual(txt, exp_txt) 