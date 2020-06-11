import unittest

from transformers import AutoTokenizer, TrainingArguments, is_torch_available

from .utils import require_torch


if is_torch_available():
    import torch
    from transformers import (
        Trainer,
        LineByLineTextDataset,
        AutoModelForSequenceClassification,
        DefaultDataCollator,
        DataCollatorForLanguageModeling,
        GlueDataset,
        GlueDataTrainingArguments,
        TextDataset,
    )


PATH_SAMPLE_TEXT = "./tests/fixtures/sample_text.txt"


@require_torch
class DataCollatorIntegrationTest(unittest.TestCase):
    def test_default_classification(self):
        MODEL_ID = "bert-base-cased-finetuned-mrpc"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
        data_collator = DefaultDataCollator()
        batch = data_collator.collate_batch(dataset.features)
        self.assertEqual(batch["labels"].dtype, torch.long)

    def test_default_regression(self):
        MODEL_ID = "distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        data_args = GlueDataTrainingArguments(
            task_name="sts-b", data_dir="./tests/fixtures/tests_samples/STS-B", overwrite_cache=True
        )
        dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
        data_collator = DefaultDataCollator()
        batch = data_collator.collate_batch(dataset.features)
        self.assertEqual(batch["labels"].dtype, torch.float)

    def test_lm_tokenizer_without_padding(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        # ^ causal lm

        dataset = LineByLineTextDataset(tokenizer, file_path=PATH_SAMPLE_TEXT, block_size=512)
        examples = [dataset[i] for i in range(len(dataset))]
        with self.assertRaises(ValueError):
            # Expect error due to padding token missing on gpt2:
            data_collator.collate_batch(examples)

        dataset = TextDataset(tokenizer, file_path=PATH_SAMPLE_TEXT, block_size=512, overwrite_cache=True)
        examples = [dataset[i] for i in range(len(dataset))]
        batch = data_collator.collate_batch(examples)
        self.assertIsInstance(batch, dict)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 512)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 512)))

    def test_lm_tokenizer_with_padding(self):
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        data_collator = DataCollatorForLanguageModeling(tokenizer)
        # ^ masked lm

        dataset = LineByLineTextDataset(tokenizer, file_path=PATH_SAMPLE_TEXT, block_size=512)
        examples = [dataset[i] for i in range(len(dataset))]
        batch = data_collator.collate_batch(examples)
        self.assertIsInstance(batch, dict)
        self.assertEqual(batch["input_ids"].shape, torch.Size((31, 107)))
        self.assertEqual(batch["labels"].shape, torch.Size((31, 107)))

        dataset = TextDataset(tokenizer, file_path=PATH_SAMPLE_TEXT, block_size=512, overwrite_cache=True)
        examples = [dataset[i] for i in range(len(dataset))]
        batch = data_collator.collate_batch(examples)
        self.assertIsInstance(batch, dict)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 512)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 512)))


@require_torch
class TrainerIntegrationTest(unittest.TestCase):
    def test_trainer_eval_mrpc(self):
        MODEL_ID = "bert-base-cased-finetuned-mrpc"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        training_args = TrainingArguments(output_dir="./examples", no_cuda=True)
        trainer = Trainer(model=model, args=training_args, eval_dataset=eval_dataset)
        result = trainer.evaluate()
        self.assertLess(result["eval_loss"], 0.2)

    def test_trainer_eval_lm(self):
        MODEL_ID = "distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer, file_path=PATH_SAMPLE_TEXT, block_size=tokenizer.max_len_single_sentence,
        )
        self.assertEqual(len(dataset), 31)
