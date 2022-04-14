import argparse
import logging
import os
import sys
import time

import tensorflow as tf
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers.utils import is_sagemaker_dp_enabled


if os.environ.get("SDP_ENABLED") or is_sagemaker_dp_enabled():
    SDP_ENABLED = True
    os.environ["SAGEMAKER_INSTANCE_TYPE"] = "p3dn.24xlarge"
    import smdistributed.dataparallel.tensorflow as sdp
else:
    SDP_ENABLED = False


def fit(model, loss, opt, train_dataset, epochs, train_batch_size, max_steps=None):
    pbar = tqdm(train_dataset)
    for i, batch in enumerate(pbar):
        with tf.GradientTape() as tape:
            inputs, targets = batch
            outputs = model(batch)
            loss_value = loss(targets, outputs.logits)

        if SDP_ENABLED:
            tape = sdp.DistributedGradientTape(tape, sparse_as_dense=True)

        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        pbar.set_description(f"Loss: {loss_value:.4f}")

        if SDP_ENABLED and i == 0:
            sdp.broadcast_variables(model.variables, root_rank=0)
            sdp.broadcast_variables(opt.variables(), root_rank=0)

        if max_steps and i >= max_steps:
            break

    train_results = {"loss": loss_value.numpy()}
    return train_results


def get_datasets(tokenizer, train_batch_size, eval_batch_size):
    # Load dataset
    train_dataset, test_dataset = load_dataset("imdb", split=["train", "test"])

    # Preprocess train dataset
    train_dataset = train_dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length"), batched=True
    )
    train_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])

    train_features = {
        x: train_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length])
        for x in ["input_ids", "attention_mask"]
    }
    tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_dataset["label"]))

    # Preprocess test dataset
    test_dataset = test_dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length"), batched=True
    )
    test_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])

    test_features = {
        x: test_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length])
        for x in ["input_ids", "attention_mask"]
    }
    tf_test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_dataset["label"]))

    if SDP_ENABLED:
        tf_train_dataset = tf_train_dataset.shard(sdp.size(), sdp.rank())
        tf_test_dataset = tf_test_dataset.shard(sdp.size(), sdp.rank())
    tf_train_dataset = tf_train_dataset.batch(train_batch_size, drop_remainder=True)
    tf_test_dataset = tf_test_dataset.batch(eval_batch_size, drop_remainder=True)

    return tf_train_dataset, tf_test_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_steps", type=int, default=None)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if SDP_ENABLED:
        sdp.init()

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[sdp.local_rank()], "GPU")

    # Load model and tokenizer
    model = TFAutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # get datasets
    tf_train_dataset, tf_test_dataset = get_datasets(
        tokenizer=tokenizer,
        train_batch_size=args.per_device_train_batch_size,
        eval_batch_size=args.per_device_eval_batch_size,
    )

    # fine optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Training
    if args.do_train:

        # train_results = model.fit(tf_train_dataset, epochs=args.epochs, batch_size=args.train_batch_size)
        start_train_time = time.time()
        train_results = fit(
            model,
            loss,
            optimizer,
            tf_train_dataset,
            args.epochs,
            args.per_device_train_batch_size,
            max_steps=args.max_steps,
        )
        end_train_time = time.time() - start_train_time
        logger.info("*** Train ***")
        logger.info(f"train_runtime = {end_train_time}")

        output_eval_file = os.path.join(args.output_dir, "train_results.txt")

        if not SDP_ENABLED or sdp.rank() == 0:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Train results *****")
                logger.info(train_results)
                for key, value in train_results.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    # Evaluation
    if args.do_eval and (not SDP_ENABLED or sdp.rank() == 0):

        result = model.evaluate(tf_test_dataset, batch_size=args.per_device_eval_batch_size, return_dict=True)
        logger.info("*** Evaluate ***")

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info(result)
            for key, value in result.items():
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    # Save result
    if SDP_ENABLED:
        if sdp.rank() == 0:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
