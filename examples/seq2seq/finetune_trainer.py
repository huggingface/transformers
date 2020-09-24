import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from seq2seq_trainer import Seq2SeqTrainer
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartTokenizer,
    EvalPrediction,
    HfArgumentParser,
    MBartTokenizer,
    T5Tokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_bart import shift_tokens_right
from utils import (
    LegacySeq2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    freeze_params,
    lmap,
    trim_batch,
    use_task_specific_params,
)


logger = logging.getLogger(__name__)


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.add_prefix_space = isinstance(tokenizer, BartTokenizer)

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            batch = self._encode(batch)
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])

            labels = trim_batch(labels, self.pad_token_id)
            input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)

        if isinstance(self.tokenizer, T5Tokenizer):
            decoder_input_ids = self._shift_right_t5(labels)
            labels = labels
        else:
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id)
            labels = labels

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
        return batch

    def _shift_right_t5(self, input_ids):
        decoder_start_token_id = self.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        return shifted_input_ids

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            src_lang=self.data_args.src_lang,
            tgt_texts=[x["tgt_texts"] for x in batch],
            tgt_lang=self.data_args.tgt_lang,
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="max_length" if self.tpu_num_cores is not None else "longest",  # TPU hack
            return_tensors="pt",
            add_prefix_space=self.add_prefix_space,
        )
        return batch_encoding.data


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        label_smoothing (:obj:`float`, `optional`, defaults to 0):
            The label smoothing epsilon to apply (if not zero).
        sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to SortishSamler or not. It sorts the inputs according to lenghts in-order to minimizing the padding size.
        predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """

    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # use task specific params
    use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if data_args.eval_beams is not None:
        model.config.num_beams = data_args.eval_beams
    assert model.config.num_beams >= 1, f"got eval_beams={model.config.num_beams}. Need an integer >= 1"

    # set max length for generation
    model.config.max_generate_length = data_args.val_max_target_length

    # set decoder_start_token_id for MBart
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        decoder_start_token_id = tokenizer.lang_code_to_id[data_args.tgt_lang]
        model.config.decoder_start_token_id = decoder_start_token_id

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def non_pad_len(tokens: np.ndarray) -> int:
            return np.count_nonzero(tokens != tokenizer.pad_token_id)

        def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
            pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
            pred_str = lmap(str.strip, pred_str)
            label_str = lmap(str.strip, label_str)
            return pred_str, label_str

        def summarization_metrics(pred: EvalPrediction) -> Dict:
            pred_str, label_str = decode_pred(pred)
            rouge: Dict = calculate_rouge(pred_str, label_str)
            summ_len = np.mean(lmap(non_pad_len, pred.predictions))
            rouge.update({"gen_len": summ_len})
            return rouge

        def translation_metrics(pred: EvalPrediction) -> Dict:
            pred_str, label_str = decode_pred(pred)
            bleu: Dict = calculate_bleu(pred_str, label_str)
            gen_len = np.mean(lmap(non_pad_len, pred.predictions))
            bleu.update({"gen_len": gen_len})
            return bleu

        compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
        return compute_metrics_fn

    def freeze_embeds(model: torch.nn.Module):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(model.model.shared)
            for d in [model.model.encoder, model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(model.shared)
            for d in [model.encoder, model.decoder]:
                freeze_params(d.embed_tokens)

    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    dataset_class = Seq2SeqDataset if hasattr(tokenizer, "prepare_seq2seq_batch") else LegacySeq2SeqDataset

    # Get datasets
    train_dataset = (
        dataset_class(
            tokenizer,
            type_path="train",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        dataset_class(
            tokenizer,
            type_path="val",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        dataset_class(
            tokenizer,
            type_path="test",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_test,
            max_target_length=data_args.test_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_predict
        else None
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Seq2SeqDataCollator(tokenizer, data_args, training_args.tpu_num_cores),
        compute_metrics=build_compute_metrics_fn(data_args.task) if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.json")
        if trainer.is_world_process_zero():
            logger.info("***** Eval results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)

            with open(output_eval_file, "w") as f:
                json.dump(result, f)

            eval_results.update(result)

    if training_args.do_predict:
        logging.info("*** Test ***")

        test_output = trainer.predict(test_dataset=test_dataset)
        test_metrics = test_output.metrics
        test_metrics = {k.replace("eval", "test"): v for k, v in test_metrics.items()}

        output_test_file = os.path.join(training_args.output_dir, "test_results.json")

        if trainer.is_world_process_zero():
            logger.info("***** Test results *****")
            for key, value in test_metrics.items():
                logger.info("  %s = %s", key, value)

            with open(output_test_file, "w") as f:
                json.dump(test_metrics, f)

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(test_output.predictions, skip_special_tokens=True)
                test_preds = lmap(str.strip, test_preds)
                output_test_pred_file = os.path.join(training_args.output_dir, "test_generations.txt")
                with open(output_test_pred_file, "w") as f:
                    f.write("\n".join(test_preds))

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
