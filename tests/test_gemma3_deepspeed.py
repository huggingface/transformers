import pytest
import torch
import json
import tempfile
import os
from transformers import (
    Gemma3ForConditionalGeneration,
    Gemma3Config,
    Trainer,
    TrainingArguments,
)
from transformers.testing_utils import (
    require_deepspeed,
    require_torch_accelerator,
    TestCasePlus,
    mockenv_context,
)
from transformers.integrations.deepspeed import (
    unset_hf_deepspeed_config,
)

# Use consistent naming and structure like existing tests
@require_deepspeed
@require_torch_accelerator  # Add this decorator
class TestGemma3DeepSpeed(TestCasePlus):  # Inherit from TestCasePlus
    """Test DeepSpeed integration with Gemma3 models"""

    def setUp(self):
        """Setup method following existing pattern"""
        super().setUp()

        # Follow the existing pattern for distributed environment
        master_port = self.get_master_port(real_launcher=False)
        self.dist_env_1_gpu = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }

    def tearDown(self):
        """Cleanup following existing pattern"""
        super().tearDown()

        # Reset deepspeed config global state
        unset_hf_deepspeed_config()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_master_port(self, real_launcher=False):
        """Helper method from existing tests"""
        master_port_base = os.environ.get("DS_TEST_PORT", "10999")
        if not real_launcher:
            master_port_base = str(int(master_port_base) + 1)
        return master_port_base

    def get_config_dict(self):
        """Create deepspeed config following existing pattern"""
        return {
            "train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "optimizer": {"type": "AdamW", "params": {"lr": 5e-5}},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
            },
            "fp16": {"enabled": True},
        }

    def get_model_config(self):
        """Create model config"""
        return Gemma3Config(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=1024,
            pad_token_id=0,
        )

    def test_gemma3_deepspeed_zero3_initialization(self):
        """Test DeepSpeed ZeRO-3 initialization with Gemma3"""
        # Use mockenv_context like existing tests
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config = self.get_config_dict()
            model_config = self.get_model_config()

            with tempfile.TemporaryDirectory() as tmp_dir:
                model = Gemma3ForConditionalGeneration(model_config)

                training_args = TrainingArguments(
                    output_dir=tmp_dir,
                    per_device_train_batch_size=1,
                    num_train_epochs=1,
                    deepspeed=ds_config,
                    logging_steps=1,
                    save_steps=999999,
                    report_to=[],
                    dataloader_num_workers=0,  # Avoid multiprocessing issues
                )

                # Create dummy dataset like existing tests
                dummy_dataset = [
                    {
                        "input_ids": torch.randint(0, 1000, (128,)),  # Smaller size
                        "labels": torch.randint(0, 1000, (128,)),
                    }
                    for _ in range(2)  # Fewer samples
                ]

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=dummy_dataset,
                )

                # Assertions following existing pattern
                self.assertIsNotNone(trainer)
                self.assertTrue(hasattr(trainer.model, 'config'))
                self.assertEqual(trainer.model.config.vocab_size, 1000)

    def test_gemma3_basic_config_creation(self):
        """Test basic model configuration without DeepSpeed"""
        model_config = self.get_model_config()
        model = Gemma3ForConditionalGeneration(model_config)

        self.assertEqual(model.config.vocab_size, 1000)
        self.assertEqual(model.config.hidden_size, 512)
        self.assertEqual(model.config.num_hidden_layers, 2)
        self.assertEqual(model.config.num_attention_heads, 8)

    @require_deepspeed
    def test_gemma3_deepspeed_stage2(self):
        """Test with DeepSpeed ZeRO Stage 2"""
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config = self.get_config_dict()
            ds_config["zero_optimization"]["stage"] = 2  # Change to stage 2

            model_config = self.get_model_config()

            with tempfile.TemporaryDirectory() as tmp_dir:
                model = Gemma3ForConditionalGeneration(model_config)

                training_args = TrainingArguments(
                    output_dir=tmp_dir,
                    per_device_train_batch_size=1,
                    max_steps=1,  # Just one step for testing
                    deepspeed=ds_config,
                    report_to=[],
                )

                dummy_dataset = [{"input_ids": torch.randint(0, 1000, (64,)), "labels": torch.randint(0, 1000, (64,))}]

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=dummy_dataset,
                )

                # Just test initialization, not training
                self.assertIsNotNone(trainer)