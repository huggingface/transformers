# This test is meant to be run in torch.distributed,
# on a machine with multiple GPUs, in the following way:
#
#   python -m torch.distributed.launch --nproc_per_node 2 ./tests/test_trainer_distributed.py
#
# Replace 2 with the number of GPUs you have.
#

import logging
import sys
from typing import Dict

from transformers import EvalPrediction, HfArgumentParser, TrainingArguments, is_torch_available


logger = logging.getLogger(__name__)


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    from transformers import DataCollator, Trainer

    class DummyDataset(Dataset):
        def __init__(self, length: int = 101):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, i) -> int:
            return i

    class DummyDataCollator(DataCollator):
        def collate_batch(self, features):
            return {"input_ids": torch.tensor(features), "labels": torch.tensor(features)}

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Add some (unused) params otherwise DDP will complain.
            self.fc = nn.Linear(120, 80)

        def forward(self, input_ids, labels=None):
            if labels is not None:
                return torch.tensor(0.0), input_ids
            else:
                return input_ids


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses(sys.argv + ["--output_dir", "./examples"])[0]

    logging.basicConfig(level=logging.INFO)
    logger.warning(
        "Process rank: %s, device: %s", training_args.local_rank, training_args.device,
    )

    for dataset_length in [101, 40, 7]:
        dataset = DummyDataset(dataset_length)

        def compute_metrics(p: EvalPrediction) -> Dict:
            sequential = list(range(len(dataset)))
            success = p.predictions.tolist() == sequential and p.label_ids.tolist() == sequential
            return {"success": success}

        trainer = Trainer(
            model=DummyModel(),
            args=training_args,
            data_collator=DummyDataCollator(),
            eval_dataset=dataset,
            compute_metrics=compute_metrics,
        )
        metrics = trainer.evaluate()
        logger.info(metrics)
        if metrics["eval_success"] is not True:
            logger.error(metrics)
            exit(1)

        p = trainer.predict(dataset)
        logger.info(p.metrics)
        if p.metrics["eval_success"] is not True:
            logger.error(p.metrics)
            exit(1)

    logger.info("🔥 All distributed tests successful")
