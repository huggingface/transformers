import unittest
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import os
import shutil

class TestCodeT5pCheckpointSaving(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-2b", trust_remote_code=True)
        self.output_dir = "./test_output"
        self.training_args = TrainingArguments(output_dir=self.output_dir, per_device_train_batch_size=1)
        self.trainer = Trainer(model=self.model, args=self.training_args)

    def test_save_model_no_assertion_error(self):
        try:
            self.trainer.save_model(self.output_dir)
            self.assertTrue(True)  # No exception raised
        except AssertionError as e:
            self.fail(f"AssertionError raised: {str(e)}")

    def test_checkpoint_contains_configs(self):
        self.trainer.save_model(self.output_dir)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "pytorch_model.bin")))

    def test_checkpoint_reloadable(self):
        self.trainer.save_model(self.output_dir)
        reloaded_model = AutoModelForSeq2SeqLM.from_pretrained(self.output_dir, trust_remote_code=True)
        self.assertIsNotNone(reloaded_model)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

if __name__ == "__main__":
    unittest.main()