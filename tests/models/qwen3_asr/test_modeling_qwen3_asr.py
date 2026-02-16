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
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


class Qwen3ASRModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 1
        self.seq_length = 10
        self.audio_token_id = 0
        self.is_training = False

        text_config = {
            "model_type": "Qwen3ASRTextConfig",
            "vocab_size": 99,   
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 16,
            "bos_token_id": 0,
            "pad_token_id": 1,
            "eos_token_id": 2,
            "decoder_start_token_id": 0,
            "tie_word_embeddings": False,
            "output_attentions": True,
            "output_hidden_states": True,
        }
        audio_config = {
            "model_type": "Qwen3ASRAudioEncoderConfig",
            "d_model": 8,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 16,
        }

        self.text_config = text_config
        self.audio_config = audio_config
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]

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
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()


@require_torch
class Qwen3ASRForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase): 
    all_model_classes = (Qwen3ASRForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {
        "automatic-speech-recognition": Qwen3ASRForConditionalGeneration,
    } if is_torch_available() else {}
    
    def setUp(self):
        self.model_tester = Qwen3ASRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3ASRConfig)

    
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
        )
        
        torch.testing.assert_close(gen_ids.cpu(), exp_ids)
        self.assertListEqual(txt, exp_txt) 