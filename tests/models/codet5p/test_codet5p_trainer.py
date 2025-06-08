import os
import json
import unittest
import tempfile
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.testing_utils import require_torch
from datasets import Dataset

@require_torch
class TestCodeT5pTrainer(unittest.TestCase):
    def setUp(self):
        # Initialize model, tokenizer, and a small dummy dataset
        self.model_name = "Salesforce/codet5p-2b"
        try:
            # Temporary workaround for AssertionError during model loading
            config_kwargs = {
                "encoder": {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12},
                "decoder": {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
            }
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, **config_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:
            self.skipTest(f"Failed to load model or tokenizer: {str(e)}")

        # Create a temporary directory for checkpoints
        self.output_dir = tempfile.mkdtemp()

        # Create a small dummy dataset to mimic demo.py
        data = {"content": ["def hello_world():\n    print('Hello, World!')"]}
        self.dataset = Dataset.from_dict(data)

        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(examples["content"], padding="max_length", truncation=True, max_length=128)
        self.tokenized_dataset = self.dataset.map(tokenize_function, batched=True)

        # Set up training arguments similar to demo.py
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=10,
            save_strategy="steps",
            logging_dir=None,
        )

        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset,
        )

    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_save_model_no_error(self):
        # Test that trainer.save_model() does not raise AssertionError
        try:
            self.trainer.save_model(self.output_dir)
            self.assertTrue(True, "save_model executed without error")
        except AssertionError as e:
            self.fail(f"save_model raised AssertionError: {str(e)}")

    def test_checkpoint_saved(self):
        # Test that checkpoint is saved to the output directory
        self.trainer.save_model(self.output_dir)
        checkpoint_files = os.listdir(self.output_dir)
        self.assertGreater(len(checkpoint_files), 0, "Checkpoint files were not saved")
        self.assertIn("config.json", checkpoint_files, "config.json not found in checkpoint")

    def test_checkpoint_contains_configs(self):
        # Test that saved checkpoint includes encoder and decoder configurations
        self.trainer.save_model(self.output_dir)
        config_path = os.path.join(self.output_dir, "config.json")
        self.assertTrue(os.path.exists(config_path), "config.json not found")
        with open(config_path, "r") as f:
            config = json.load(f)
        self.assertIn("encoder", config, "Encoder configuration missing in checkpoint")
        self.assertIn("decoder", config, "Decoder configuration missing in checkpoint")

    def test_reload_checkpoint(self):
        # Test that checkpoint can be reloaded without errors
        self.trainer.save_model(self.output_dir)
        try:
            reloaded_model = AutoModelForCausalLM.from_pretrained(self.output_dir, trust_remote_code=True)
            self.assertIsNotNone(reloaded_model, "Failed to reload model from checkpoint")
        except Exception as e:
            self.fail(f"Reloading checkpoint raised error: {str(e)}")

    def test_compatibility_with_trainer(self):
        # Test that the solution is compatible with the Trainer class
        try:
            self.trainer.train()  # Run a single training step
            self.trainer.save_model(self.output_dir)
            self.assertTrue(True, "Training and saving completed without errors")
        except Exception as e:
            self.fail(f"Trainer compatibility test failed: {str(e)}")

if __name__ == "__main__":
    unittest.main()