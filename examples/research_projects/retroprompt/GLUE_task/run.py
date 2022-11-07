import logging
import os
import warnings
from datetime import datetime

import numpy as np
import torch

from src.compute_if import compute_memorize_score
from src.data_processor import bound_mapping, num_labels_mapping, output_modes_mapping
from src.dataset import FewShotDataset
from src.models import RobertaForPromptFinetuning
from src.trainer import Trainer
from src.training_args import DynamicDataTrainingArguments, DynamicTrainingArguments, ModelArguments
from src.utils import build_compute_metrics_fn, data_collator_for_cl
from transformers import AutoConfig, AutoModelForSequenceClassification, HfArgumentParser, RobertaTokenizer, set_seed


warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
DEVICE = "cuda"
os.environ["WANDB_DISABLED"] = "true"


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args.knn_mode = training_args.knn_mode
    model_args.train_with_knn = training_args.train_with_knn
    model_args.only_train_knn = training_args.only_train_knn
    data_args.use_demo = training_args.use_demo
    if "prompt" in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info(
            "Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode)
        )
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
    )

    special_tokens = []
    if config.model_type == "roberta":
        tokenizer_fn = RobertaTokenizer
    else:
        raise NotImplementedError

    tokenizer = tokenizer_fn.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
    )

    data_args.output_dir = training_args.output_dir
    train_dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="train")
    eval_dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="dev")

    test_dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="test") if training_args.do_predict else None

    # add some parameters to generate specific model.
    config.virtual_demo = data_args.virtual_demo
    config.virtual_demo_length_per_label = data_args.virtual_demo_length_per_label
    config.virtual_demo_init = data_args.virtual_demo_init
    config.total_label_tokens = train_dataset.total_label_tokens

    if "prompt" in model_args.few_shot_type:
        if config.model_type == "roberta":
            model_fn = RobertaForPromptFinetuning
        else:
            raise NotImplementedError
    elif model_args.few_shot_type == "finetune":
        model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError

    model = model_fn.from_pretrained(
        model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path), config=config,
    )

    if data_args.prompt:
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().to(DEVICE)
    if output_modes_mapping[data_args.task_name] == "regression":
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]

    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        data_collator=data_collator_for_cl,
    )

    # Training
    if training_args.do_train:
        if training_args.ckpt_dir is not None:  # transfer continual
            model = model_fn.from_pretrained(training_args.ckpt_dir)
            model.data_args = data_args
            model.model_args = model_args
            model.tokenizer = tokenizer

            model.to(DEVICE)
            if data_args.prompt:
                model.label_word_list = torch.tensor(train_dataset.label_word_list).long().to(DEVICE)
            if output_modes_mapping[data_args.task_name] == "regression":
                # lower / upper bounds
                model.lb, model.ub = bound_mapping[data_args.task_name]

            trainer.model = model

        trainer.train()

        tokenizer.save_pretrained(training_args.output_dir)
        torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
        torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))

        model = model_fn.from_pretrained(training_args.output_dir)
        model.data_args = data_args
        model.model_args = model_args
        model.tokenizer = tokenizer

        model.to(DEVICE)

        if data_args.prompt:
            model.label_word_list = torch.tensor(train_dataset.label_word_list).long().to(DEVICE)
        if output_modes_mapping[data_args.task_name] == "regression":
            # lower / upper bounds
            model.lb, model.ub = bound_mapping[data_args.task_name]

        trainer.model = model

    final_result = {
        "time": str(datetime.today()),
    }
    if not training_args.do_train and not training_args.do_case:
        if training_args.ckpt_dir is not None:
            model = model_fn.from_pretrained(training_args.ckpt_dir)
        elif training_args.output_dir is not None:
            model = model_fn.from_pretrained(training_args.output_dir)
        model.data_args = data_args
        model.model_args = model_args
        model.tokenizer = tokenizer

        model.to(DEVICE)
        if data_args.prompt:
            model.label_word_list = torch.tensor(train_dataset.label_word_list).long().to(DEVICE)
        if output_modes_mapping[data_args.task_name] == "regression":
            # lower / upper bounds
            model.lb, model.ub = bound_mapping[data_args.task_name]

        trainer.model = model

        if training_args.compute_mem:
            # compute memorize scores and exit
            logger.info("Compute memorize...")
            compute_memorize_score(trainer)
            exit(0)

    if training_args.do_eval:
        logger.info("**** Evaluation ****")

        trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
        output = trainer.evaluate()
        eval_result = output.metrics

        output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
                final_result[eval_dataset.args.task_name + "_dev_" + key] = value

    test_results = {}
    if training_args.do_predict:
        logger.info("**** Test ****")

        test_datasets = [test_dataset]
        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}_infer.txt"
            )

            with open(output_test_file, "w") as writer:
                logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                for key, value in test_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
                    final_result[test_dataset.args.task_name + "_test_" + key] = value

            if training_args.save_logit:
                predictions = output.predictions
                num_logits = predictions.shape[-1]
                logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                np.save(
                    os.path.join(
                        training_args.save_logit_dir,
                        "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id),
                    ),
                    logits,
                )
            test_results.update(test_result)


if __name__ == "__main__":
    main()
