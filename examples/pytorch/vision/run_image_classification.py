import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# from datasets import load_dataset
import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_dir: str = field(
        metadata={"help": "Path to the root training directory which contains one subdirectory per class."}
    )
    validation_dir: str = field(
        default=None,
        metadata={
            "help": "Path to the root validation directory which contains one subdirectory per class. If not provided, the validation dataset will be split off of training dataset using train_val_split"
        },
    )
    test_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the test data."})
    train_val_split: Optional[float] = field(
        default=0.15,
        metadata={
            "help": "Percent to split off of train for validation. Only splits if validation_dir is not provided."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    image_size: Optional[int] = field(default=224, metadata={"help": " The size (resolution) of each image."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors="pt")
        encodings["labels"] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_dataset = ImageFolder(
        data_args.train_dir,
        T.Compose(
            [
                T.RandomResizedCrop(data_args.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ]
        ),
    )
    labels = train_dataset.classes
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    if data_args.validation_dir is not None:
        eval_dataset = ImageFolder(
            data_args.validation_dir,
            T.Compose(
                [
                    T.Resize(data_args.image_size),
                    T.CenterCrop(data_args.image_size),
                    T.ToTensor(),
                    normalize,
                ]
            ),
        )
    else:
        indices = torch.randperm(len(train_dataset)).tolist()
        n_val = math.floor(len(indices) * data_args.train_val_split)
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:-n_val])
        eval_dataset = torch.utils.data.Subset(train_dataset, indices[-n_val:])

    test_dataset = (
        ImageFolder(
            data_args.test_dir,
            T.Compose(
                [
                    T.Resize(data_args.image_size),
                    T.CenterCrop(data_args.image_size),
                    T.ToTensor(),
                    normalize,
                ]
            ),
        )
        if data_args.test_dir is not None
        else None
    )

    metric = datasets.load_metric("accuracy")

    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    def model_init():
        return AutoModelForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # TODO - Limit samples for dataset splits here based on values provided to data_args

    # TODO - Make sure we are using the feature extractor correctly in relation to previously defined train/val transforms
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=ImageClassificationCollator(feature_extractor),
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predictions = trainer.predict(test_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Writing predict results *****")
                writer.write("index\tfile\tprediction\n")
                for i, prediction in enumerate(predictions):
                    file = test_dataset.imgs[i][0].split("/")[-1]
                    writer.write(f"{i}\t{file}\t{prediction}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "image-classification"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
