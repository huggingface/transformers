# coding=utf-8
import collections
import datetime
import glob
import math
import os
import re

import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from seqeval import metrics

from transformers import (
    TF2_WEIGHTS_NAME,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoTokenizer,
    GradientAccumulator,
    PreTrainedTokenizer,
    TFAutoModelForTokenClassification,
    create_optimizer,
)
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file


MODEL_CONFIG_CLASSES = list(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())


flags.DEFINE_string(
    "data_dir", None, "The input data dir. Should contain the .conll files (or other data files) for the task."
)

flags.DEFINE_string(
    "model_name_or_path", None, "Path to pretrained model or model identifier from huggingface.co/models",
)

flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "labels", "", "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."
)

flags.DEFINE_string("config_name", None, "Pretrained config name or path if not the same as model_name")

flags.DEFINE_string("tokenizer_name", None, "Pretrained tokenizer name or path if not the same as model_name")

flags.DEFINE_string("cache_dir", None, "Where do you want to store the pre-trained models downloaded from s3")

flags.DEFINE_integer(
    "max_seq_length",
    128,
    "The maximum total input sentence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter "
    "will be padded.",
)

flags.DEFINE_boolean("do_train", False, "Whether to run training.")

flags.DEFINE_boolean("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_boolean("do_predict", False, "Whether to run predictions on the test set.")

flags.DEFINE_boolean(
    "evaluate_during_training", False, "Whether to run evaluation during training at each logging step."
)

flags.DEFINE_boolean("do_lower_case", False, "Set this flag if you are using an uncased model.")

flags.DEFINE_integer("per_device_train_batch_size", 8, "Batch size per GPU/CPU/TPU for training.")

flags.DEFINE_integer("per_device_eval_batch_size", 8, "Batch size per GPU/CPU/TPU for evaluation.")

flags.DEFINE_integer(
    "gradient_accumulation_steps", 1, "Number of updates steps to accumulate before performing a backward/update pass."
)

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("weight_decay", 0.0, "Weight decay if we apply some.")

flags.DEFINE_float("adam_epsilon", 1e-8, "Epsilon for Adam optimizer.")

flags.DEFINE_float("max_grad_norm", 1.0, "Max gradient norm.")

flags.DEFINE_integer("num_train_epochs", 3, "Total number of training epochs to perform.")

flags.DEFINE_integer(
    "max_steps", -1, "If > 0: set total number of training steps to perform. Override num_train_epochs."
)

flags.DEFINE_integer("warmup_steps", 0, "Linear warmup over warmup_steps.")

flags.DEFINE_integer("logging_steps", 50, "Log every X updates steps.")

flags.DEFINE_integer("save_steps", 50, "Save checkpoint every X updates steps.")

flags.DEFINE_boolean(
    "eval_all_checkpoints",
    False,
    "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)

flags.DEFINE_boolean("no_cuda", False, "Avoid using CUDA even if it is available")

flags.DEFINE_boolean("overwrite_output_dir", False, "Overwrite the content of the output directory")

flags.DEFINE_boolean("overwrite_cache", False, "Overwrite the cached training and evaluation sets")

flags.DEFINE_integer("seed", 42, "random seed for initialization")

flags.DEFINE_boolean("fp16", False, "Whether to use 16-bit (mixed) precision instead of 32-bit")

flags.DEFINE_string(
    "gpus",
    "0",
    "Comma separated list of gpus devices. If only one, switch to single "
    "gpu strategy, if None takes all the gpus available.",
)


def run_model(train_features, train_labels, training, model, labels, pad_token_label_id, loss_fct):
    """
    Computes the loss of the given features and labels pair.
    Args:
        train_features: the batched features.
        train_labels: the batched labels.
    """
    logits = model(train_features, training=training)[0]
    active_loss = tf.reshape(train_labels, (-1,)) != pad_token_label_id
    active_logits = tf.boolean_mask(tf.reshape(logits, (-1, len(labels))), active_loss)
    active_labels = tf.boolean_mask(tf.reshape(train_labels, (-1,)), active_loss)

    loss = loss_fct(active_labels, active_logits)

    return loss, logits


def train(
    args,
    strategy,
    train_dataset,
    tokenizer,
    model,
    num_train_examples,
    labels,
    train_batch_size,
    pad_token_label_id,
    model_type,
):
    if args["max_steps"] > 0:
        num_train_steps = args["max_steps"]
        args["num_train_epochs"] = 1
    else:
        num_train_steps = math.ceil(num_train_examples / train_batch_size)

    writer = tf.summary.create_file_writer("/tmp/mylogs")

    with strategy.scope():
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        optimizer = create_optimizer(args["learning_rate"], num_train_steps, args["warmup_steps"])

        if args["fp16"]:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

        gradient_accumulator = GradientAccumulator()

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", num_train_examples)
    logging.info("  Num Epochs = %d", args["num_train_epochs"])
    logging.info("  Instantaneous batch size per device = %d", args["per_device_train_batch_size"])
    logging.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size,
    )
    logging.info("  Gradient Accumulation steps = %d", args["gradient_accumulation_steps"])
    logging.info("  Total training steps = %d", num_train_steps)

    model.summary()

    def training_steps():
        """
        Returns a generator over training steps (i.e. parameters update).
        Args:
          dataset: The training dataset.
        Returns:
          A generator that yields a loss value to report for this step.
        """
        for i, loss in enumerate(accumulate_next_gradients()):
            if i % args["gradient_accumulation_steps"] == 0:
                apply_gradients()
                yield loss

    @tf.function
    def apply_gradients():
        """Applies the gradients (cross-replica)."""
        strategy.experimental_run_v2(step)

    def step():
        """Applies gradients and resets accumulation."""
        gradient_scale = gradient_accumulator.step * strategy.num_replicas_in_sync
        gradients = [gradient / tf.cast(gradient_scale, gradient.dtype) for gradient in gradient_accumulator.gradients]
        gradients = [(tf.clip_by_value(grad, -args["max_grad_norm"], args["max_grad_norm"])) for grad in gradients]
        vars = [var for var in model.trainable_variables if "pooler" not in var.name]

        optimizer.apply_gradients(list(zip(gradients, vars)))
        gradient_accumulator.reset()

    def accumulate_next_gradients():
        """Accumulates the gradients from the next element in dataset."""
        iterator = iter(train_dataset)

        @tf.function
        def accumulate_next():
            per_replica_features, per_replica_labels = next(iterator)

            return accumulate_gradients(per_replica_features, per_replica_labels)

        while True:
            try:
                yield accumulate_next()
            except tf.errors.OutOfRangeError:
                break

    def accumulate_gradients(per_replica_features, per_replica_labels):
        """Accumulates the gradients across all the replica."""
        per_replica_loss = strategy.experimental_run_v2(forward, args=(per_replica_features, per_replica_labels))

        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

    def forward(per_replica_features, per_replica_labels):
        """Forwards a training example and accumulates the gradients."""
        per_example_loss, logits = run_model(
            per_replica_features, per_replica_labels, True, model, labels, pad_token_label_id, loss_fct
        )
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=train_batch_size)
        vars = [var for var in model.trainable_variables if "pooler" not in var.name]
        gradients = optimizer.get_gradients(loss, vars)

        gradient_accumulator(gradients)

        return per_example_loss

    current_time = datetime.datetime.now()
    global_step = 0
    iterations = optimizer.iterations

    for epoch in range(1, args["num_train_epochs"] + 1):
        for training_loss in training_steps():
            global_step = iterations.numpy()
            training_loss = tf.reduce_mean(training_loss)

            if args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                logging.info("Epoch {} Step {} Loss {:.4f}".format(epoch, global_step, training_loss.numpy()))
                # Log metrics
                if args["evaluate_during_training"]:
                    y_true, y_pred, eval_loss = evaluate(
                        args, strategy, model, tokenizer, labels, pad_token_label_id, model_type, mode="dev"
                    )
                    report = metrics.classification_report(y_true, y_pred, digits=4)

                    logging.info("Eval at step " + str(global_step) + "\n" + report)
                    logging.info("eval_loss: " + str(eval_loss))

                    precision = metrics.precision_score(y_true, y_pred)
                    recall = metrics.recall_score(y_true, y_pred)
                    f1 = metrics.f1_score(y_true, y_pred)

                    with writer.as_default():
                        tf.summary.scalar("eval_loss", eval_loss, global_step)
                        tf.summary.scalar("precision", precision, global_step)
                        tf.summary.scalar("recall", recall, global_step)
                        tf.summary.scalar("f1", f1, global_step)

                lr = optimizer.learning_rate
                learning_rate = lr(global_step)

                with writer.as_default():
                    tf.summary.scalar("lr", learning_rate, global_step)
                    tf.summary.scalar("loss", training_loss, global_step)

            with writer.as_default():
                tf.summary.scalar("loss", training_loss, step=global_step)

            if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                # Save model checkpoint
                output_dir = os.path.join(args["output_dir"], "checkpoint-{}".format(global_step))

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model.save_pretrained(output_dir)
                logging.info("Saving model checkpoint to %s", output_dir)

            if global_step % num_train_steps == 0:
                break

    logging.info("  Training took time = {}".format(datetime.datetime.now() - current_time))


def evaluate(args, strategy, model, tokenizer, labels, pad_token_label_id, model_type, mode):
    eval_batch_size = args["per_device_eval_batch_size"]
    eval_dataset, size = load_and_cache_examples(
        args, tokenizer, labels, pad_token_label_id, eval_batch_size, model_type, mode=mode
    )
    eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)
    preds = None
    num_eval_steps = math.ceil(size / eval_batch_size)
    loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = 0.0

    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", size)
    logging.info("  Batch size = %d", eval_batch_size)
    step = 1

    for eval_features, eval_labels in eval_dataset:
        loss, logits = run_model(eval_features, eval_labels, False, model, labels, pad_token_label_id, loss_fct)
        loss = tf.reduce_mean(loss)

        if preds is None:
            preds = logits.numpy()
            label_ids = eval_labels.numpy()
        else:
            preds = np.append(preds, logits.numpy(), axis=0)
            label_ids = np.append(label_ids, eval_labels.numpy(), axis=0)

        if step == num_eval_steps:
            break

        step += 1

    preds = np.argmax(preds, axis=2)
    y_pred = [[] for _ in range(label_ids.shape[0])]
    y_true = [[] for _ in range(label_ids.shape[0])]

    for i in range(label_ids.shape[0]):
        for j in range(label_ids.shape[1]):
            if label_ids[i, j] != pad_token_label_id:
                y_pred[i].append(labels[preds[i, j]])
                y_true[i].append(labels[label_ids[i, j]])

    return y_true, y_pred, loss.numpy()


def load_cache(cached_file, tokenizer: PreTrainedTokenizer, max_seq_length):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "attention_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
    }
    # TODO Find a cleaner way to do this.
    if "token_type_ids" in tokenizer.model_input_names:
        name_to_features["token_type_ids"] = tf.io.FixedLenFeature([max_seq_length], tf.int64)

    def _decode_record(record):
        example = tf.io.parse_single_example(record, name_to_features)
        features = {}
        features["input_ids"] = example["input_ids"]
        features["attention_mask"] = example["attention_mask"]
        if "token_type_ids" in example:
            features["token_type_ids"] = example["token_type_ids"]

        return features, example["label_ids"]

    d = tf.data.TFRecordDataset(cached_file)
    d = d.map(_decode_record, num_parallel_calls=4)
    count = d.reduce(0, lambda x, _: x + 1)

    return d, count.numpy()


def save_cache(features, cached_features_file):
    writer = tf.io.TFRecordWriter(cached_features_file)

    for (ex_index, feature) in enumerate(features):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(features)))

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        record_feature = collections.OrderedDict()
        record_feature["input_ids"] = create_int_feature(feature.input_ids)
        record_feature["attention_mask"] = create_int_feature(feature.attention_mask)
        if feature.token_type_ids is not None:
            record_feature["token_type_ids"] = create_int_feature(feature.token_type_ids)
        record_feature["label_ids"] = create_int_feature(feature.label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=record_feature))

        writer.write(tf_example.SerializeToString())

    writer.close()


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, batch_size, model_type, mode):
    drop_remainder = True if mode == "train" else False

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args["data_dir"],
        "cached_{}_{}_{}.tf_record".format(mode, tokenizer.__class__.__name__, str(args["max_seq_length"])),
    )
    if os.path.exists(cached_features_file) and not args["overwrite_cache"]:
        logging.info("Loading features from cached file %s", cached_features_file)
        dataset, size = load_cache(cached_features_file, tokenizer, args["max_seq_length"])
    else:
        logging.info("Creating features from dataset file at %s", args["data_dir"])
        examples = read_examples_from_file(args["data_dir"], mode)
        features = convert_examples_to_features(
            examples,
            labels,
            args["max_seq_length"],
            tokenizer,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=pad_token_label_id,
        )
        logging.info("Saving features into cached file %s", cached_features_file)
        save_cache(features, cached_features_file)
        dataset, size = load_cache(cached_features_file, tokenizer, args["max_seq_length"])

    if mode == "train":
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=8192, seed=args["seed"])

    dataset = dataset.batch(batch_size, drop_remainder)
    dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset, size


def main(_):
    logging.set_verbosity(logging.INFO)
    args = flags.FLAGS.flag_values_dict()

    if (
        os.path.exists(args["output_dir"])
        and os.listdir(args["output_dir"])
        and args["do_train"]
        and not args["overwrite_output_dir"]
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args["output_dir"]
            )
        )

    if args["fp16"]:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    if args["no_cuda"]:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:" + args["gpus"].split(",")[0])

    labels = get_labels(args["labels"])
    num_labels = len(labels)
    pad_token_label_id = -1
    config = AutoConfig.from_pretrained(
        args["config_name"] if args["config_name"] else args["model_name_or_path"],
        num_labels=num_labels,
        cache_dir=args["cache_dir"],
    )

    logging.info("Training/evaluation parameters %s", args)
    args["model_type"] = config.model_type

    # Training
    if args["do_train"]:
        tokenizer = AutoTokenizer.from_pretrained(
            args["tokenizer_name"] if args["tokenizer_name"] else args["model_name_or_path"],
            do_lower_case=args["do_lower_case"],
            cache_dir=args["cache_dir"],
        )

        with strategy.scope():
            model = TFAutoModelForTokenClassification.from_pretrained(
                args["model_name_or_path"],
                from_pt=bool(".bin" in args["model_name_or_path"]),
                config=config,
                cache_dir=args["cache_dir"],
            )

        train_batch_size = args["per_device_train_batch_size"]
        train_dataset, num_train_examples = load_and_cache_examples(
            args, tokenizer, labels, pad_token_label_id, train_batch_size, config.model_type, mode="train"
        )
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        train(
            args,
            strategy,
            train_dataset,
            tokenizer,
            model,
            num_train_examples,
            labels,
            train_batch_size,
            pad_token_label_id,
            config.model_type,
        )

        os.makedirs(args["output_dir"], exist_ok=True)

        logging.info("Saving model to %s", args["output_dir"])

        model.save_pretrained(args["output_dir"])
        tokenizer.save_pretrained(args["output_dir"])

    # Evaluation
    if args["do_eval"]:
        tokenizer = AutoTokenizer.from_pretrained(args["output_dir"], do_lower_case=args["do_lower_case"])
        checkpoints = []
        results = []

        if args["eval_all_checkpoints"]:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args["output_dir"] + "/**/" + TF2_WEIGHTS_NAME, recursive=True),
                    key=lambda f: int("".join(filter(str.isdigit, f)) or -1),
                )
            )

        logging.info("Evaluate the following checkpoints: %s", checkpoints)

        if len(checkpoints) == 0:
            checkpoints.append(args["output_dir"])

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if re.match(".*checkpoint-[0-9]", checkpoint) else "final"

            with strategy.scope():
                model = TFAutoModelForTokenClassification.from_pretrained(checkpoint)

            y_true, y_pred, eval_loss = evaluate(
                args, strategy, model, tokenizer, labels, pad_token_label_id, config.model_type, mode="dev"
            )
            report = metrics.classification_report(y_true, y_pred, digits=4)

            if global_step:
                results.append({global_step + "_report": report, global_step + "_loss": eval_loss})

        output_eval_file = os.path.join(args["output_dir"], "eval_results.txt")

        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            for res in results:
                for key, val in res.items():
                    if "loss" in key:
                        logging.info(key + " = " + str(val))
                        writer.write(key + " = " + str(val))
                        writer.write("\n")
                    else:
                        logging.info(key)
                        logging.info("\n" + report)
                        writer.write(key + "\n")
                        writer.write(report)
                        writer.write("\n")

    if args["do_predict"]:
        tokenizer = AutoTokenizer.from_pretrained(args["output_dir"], do_lower_case=args["do_lower_case"])
        model = TFAutoModelForTokenClassification.from_pretrained(args["output_dir"])
        eval_batch_size = args["per_device_eval_batch_size"]
        predict_dataset, _ = load_and_cache_examples(
            args, tokenizer, labels, pad_token_label_id, eval_batch_size, config.model_type, mode="test"
        )
        y_true, y_pred, pred_loss = evaluate(
            args, strategy, model, tokenizer, labels, pad_token_label_id, config.model_type, mode="test"
        )
        output_test_results_file = os.path.join(args["output_dir"], "test_results.txt")
        output_test_predictions_file = os.path.join(args["output_dir"], "test_predictions.txt")
        report = metrics.classification_report(y_true, y_pred, digits=4)

        with tf.io.gfile.GFile(output_test_results_file, "w") as writer:
            report = metrics.classification_report(y_true, y_pred, digits=4)

            logging.info("\n" + report)

            writer.write(report)
            writer.write("\n\nloss = " + str(pred_loss))

        with tf.io.gfile.GFile(output_test_predictions_file, "w") as writer:
            with tf.io.gfile.GFile(os.path.join(args["data_dir"], "test.txt"), "r") as f:
                example_id = 0

                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)

                        if not y_pred[example_id]:
                            example_id += 1
                    elif y_pred[example_id]:
                        output_line = line.split()[0] + " " + y_pred[example_id].pop(0) + "\n"
                        writer.write(output_line)
                    else:
                        logging.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model_name_or_path")
    app.run(main)
