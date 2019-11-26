# coding=utf-8
import datetime
import os

import tensorflow as tf
import collections
import numpy as np
from seqeval import metrics
import _pickle as pickle
from absl import logging
from transformers import BertConfig, BertTokenizer, TFBertForTokenClassification
from transformers import RobertaConfig, RobertaTokenizer, TFRobertaForTokenClassification
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForTokenClassification
from transformers import create_optimizer
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file
from fastprogress import master_bar, progress_bar
from absl import flags
from absl import app
import re


ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, DistilBertConfig)),
    ())

MODEL_CLASSES = {
    "bert": (BertConfig, TFBertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, TFRobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, TFDistilBertForTokenClassification, DistilBertTokenizer)
}


flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .conll files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "model_type", None,
    "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

flags.DEFINE_string(
    "model_name_or_path", None,
    "Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "labels", "",
    "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")

flags.DEFINE_string(
    "config_name", "",
    "Pretrained config name or path if not the same as model_name")

flags.DEFINE_string(
    "tokenizer_name", "",
    "Pretrained tokenizer name or path if not the same as model_name")

flags.DEFINE_string(
    "cache_dir", "",
    "Where do you want to store the pre-trained models downloaded from s3")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sentence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter "
    "will be padded.")

flags.DEFINE_string(
    "tpu", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Total number of TPU cores to use.")

flags.DEFINE_boolean(
    "do_train", False,
    "Whether to run training.")

flags.DEFINE_boolean(
    "do_eval", False,
    "Whether to run eval on the dev set.")

flags.DEFINE_boolean(
    "do_predict", False,
    "Whether to run predictions on the test set.")

flags.DEFINE_boolean(
    "evaluate_during_training", False,
    "Whether to run evaluation during training at each logging step.")

flags.DEFINE_boolean(
    "do_lower_case", False,
    "Set this flag if you are using an uncased model.")

flags.DEFINE_integer(
    "per_gpu_train_batch_size", 32,
    "Batch size per GPU/CPU for training.")

flags.DEFINE_integer(
    "per_gpu_eval_batch_size", 32,
    "Batch size per GPU/CPU for evaluation.")

flags.DEFINE_float(
    "learning_rate", 5e-5,
    "The initial learning rate for Adam.")

flags.DEFINE_float(
    "weight_decay", 0.0,
    "Weight decay if we apply some.")

flags.DEFINE_float(
    "adam_epsilon", 1e-8,
    "Epsilon for Adam optimizer.")

flags.DEFINE_integer(
    "num_train_epochs", 3,
    "Total number of training epochs to perform.")

flags.DEFINE_integer(
    "warmup_steps", 0,
    "Linear warmup over warmup_steps.")

flags.DEFINE_boolean(
    "overwrite_output_dir", False,
    "Overwrite the content of the output directory")

flags.DEFINE_boolean(
    "overwrite_cache", False,
    "Overwrite the cached training and evaluation sets")

flags.DEFINE_integer(
    "seed", 42,
    "random seed for initialization")

flags.DEFINE_boolean(
    "fp16", False,
    "Whether to use 16-bit (mixed) precision instead of 32-bit")

flags.DEFINE_string(
    "gpus", "0",
    "Comma separated list of gpus devices. If only one, switch to single "
    "gpu strategy, if None takes all the gpus available.")


def train(args, strategy, train_dataset, model, train_number_examples, num_labels, batch_size):
    num_train_optimization_steps = train_number_examples * args['num_train_epochs']
    num_train_steps = int(train_number_examples // batch_size)
    num_warmup_steps = int(args['warmup_steps'] * num_train_optimization_steps)

    with strategy.scope():
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        optimizer = create_optimizer(args['learning_rate'], num_train_optimization_steps, num_warmup_steps)

        if args['fp16']:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
        
        loss_metric = tf.keras.metrics.Mean()
    
    model.summary()

    @tf.function
    def train_step(features, labels):
        def step_fn(features, labels):
            inputs = {'attention_mask': features['input_mask'], 'training': True}

            if args['model_type'] != "distilbert":
                inputs["token_type_ids"] = features['segment_ids'] if args['model_type'] in ["bert", "xlnet"] else None

            with tf.GradientTape() as tape:
                logits = model(features['input_ids'], **inputs)[0]
                logits = tf.reshape(logits,(-1, num_labels))
                active_loss = tf.reshape(features['input_mask'], (-1,))
                active_logits = tf.boolean_mask(logits, active_loss)
                labels = tf.reshape(labels,(-1,))
                active_labels = tf.boolean_mask(labels, active_loss)
                cross_entropy = loss_fct(active_labels, active_logits)
                loss = tf.reduce_sum(cross_entropy) * (1.0 / batch_size)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

            return cross_entropy

        per_example_losses = strategy.experimental_run_v2(step_fn, args=(features, labels))
        mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)

        return mean_loss

    epoch_bar = master_bar(range(args['num_train_epochs']))
    current_time = datetime.datetime.now()

    for epoch in epoch_bar:
        with strategy.scope():
            for (features, labels) in progress_bar(train_dataset, total=num_train_steps, parent=epoch_bar):
                loss = train_step(features, labels)
                loss_metric(loss)
                epoch_bar.child.comment = f'loss : {loss_metric.result()}'

            epoch_bar.write(f'loss epoch {epoch + 1}: {loss_metric.result()}')

            loss_metric.reset_states()

    logging.info("  Training took time = {}".format(datetime.datetime.now() - current_time))



def evaluate(args, model, labels_list, eval_dataset, pad_token_label_id):
    preds = None

    for features, labels in eval_dataset:
        inputs = {'attention_mask': features['input_mask'], 'training': False}

        if args['model_type'] != "distilbert":
            inputs["token_type_ids"] = features['segment_ids'] if args['model_type'] in ["bert", "xlnet"] else None
        
        logits = model(features['input_ids'], **inputs)[0]

        if preds is None:
            preds = logits.numpy()
            label_ids = labels.numpy()
        else:
            preds = np.append(preds, logits.numpy(), axis=0)
            label_ids = np.append(label_ids, labels.numpy(), axis=0)

    preds = np.argmax(preds, axis=2)
    y_pred = [[] for _ in range(label_ids.shape[0])]
    y_true = [[] for _ in range(label_ids.shape[0])]

    for i in range(label_ids.shape[0]):
        for j in range(label_ids.shape[1]):
            if label_ids[i, j] != pad_token_label_id:
                y_pred[i].append(labels_list[preds[i, j] - 1])
                y_true[i].append(labels_list[label_ids[i, j] - 1])

    return y_true, y_pred


def load_cache(cached_file, max_seq_length):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
    }

    def _decode_record(record):
        example = tf.io.parse_single_example(record, name_to_features)
        features = {}
        features['input_ids'] = example['input_ids']
        features['input_mask'] = example['input_mask']
        features['segment_ids'] = example['segment_ids']
        
        return features, example['label_ids']

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
        record_feature["input_mask"] = create_int_feature(feature.input_mask)
        record_feature["segment_ids"] = create_int_feature(feature.segment_ids)
        record_feature["label_ids"] = create_int_feature(feature.label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=record_feature))
        
        writer.write(tf_example.SerializeToString())

    writer.close()


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, batch_size, mode):
    drop_remainder = True if args['tpu'] or mode == 'train' else False
    
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args['data_dir'], "cached_{}_{}_{}.tf_record".format(mode,
        list(filter(None, args['model_name_or_path'].split("/"))).pop(),
        str(args['max_seq_length'])))
    if os.path.exists(cached_features_file) and not args['overwrite_cache']:
        logging.info("Loading features from cached file %s", cached_features_file)
        dataset, size = load_cache(cached_features_file, args['max_seq_length'])
    else:
        logging.info("Creating features from dataset file at %s", args['data_dir'])
        examples = read_examples_from_file(args['data_dir'], mode)
        features = convert_examples_to_features(examples, labels, args['max_seq_length'], tokenizer,
                                                cls_token_at_end=bool(args['model_type'] in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args['model_type'] in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args['model_type'] in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args['model_type'] in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args['model_type'] in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id
                                                )
        logging.info("Saving features into cached file %s", cached_features_file)
        save_cache(features, cached_features_file)
        dataset, size = load_cache(cached_features_file, args['max_seq_length'])

    if mode == 'train':
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=8192, seed=args['seed'])

    dataset = dataset.batch(batch_size, drop_remainder)
    dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset, size


def main(_):
    logging.set_verbosity(logging.INFO)
    args = flags.FLAGS.flag_values_dict()

    if os.path.exists(args['output_dir']) and os.listdir(
            args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args['output_dir']))
    
    if args['fp16']:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    strategy = None
    args['n_gpu'] = []

    if args['tpu']:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args['tpu'])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif len(args['gpus'].split(',')) > 1:
        args['n_gpu'] = [f"/gpu:{gpu}" for gpu in args['gpus'].split(',')]
        strategy = tf.distribute.MirroredStrategy(devices=args['n_gpu'])
    else:
        args['n_gpu'] = args['gpus'].split(',')
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:" + args['n_gpu'][0])
    
    logging.warning("n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args['n_gpu'], bool(len(args['n_gpu']) > 1), args['fp16'])
    
    labels = get_labels(args['labels'])
    num_labels = len(labels) + 1
    pad_token_label_id = 0
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
    config = config_class.from_pretrained(args['config_name'] if args['config_name'] else args['model_name_or_path'],
                                          num_labels=num_labels,
                                          cache_dir=args['cache_dir'] if args['cache_dir'] else None)
    tokenizer = tokenizer_class.from_pretrained(args['tokenizer_name'] if args['tokenizer_name'] else args['model_name_or_path'],
                                                do_lower_case=args['do_lower_case'],
                                                cache_dir=args['cache_dir'] if args['cache_dir'] else None)
    
    with strategy.scope():
        model = model_class.from_pretrained(args['model_name_or_path'],
                                            from_pt=bool(".bin" in args['model_name_or_path']),
                                            config=config,
                                            cache_dir=args['cache_dir'] if args['cache_dir'] else None)
        model.layers[-1].activation = tf.keras.activations.softmax

    logging.info("Training/evaluation parameters %s", args)

    # Training
    if args['do_train']:
        train_batch_size = args['per_gpu_train_batch_size'] * max(1, len(args['n_gpu']))
        train_dataset, num_train_example = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, train_batch_size, mode="train")
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        train(args, strategy, train_dataset, model, num_train_example, num_labels, train_batch_size)

        if not os.path.exists(args['output_dir']):
            os.makedirs(args['output_dir'])
        
        logging.info("Saving model to %s", args['output_dir'])
        
        model.save_pretrained(args['output_dir'])
        tokenizer.save_pretrained(args['output_dir'])
    
    # Evaluation
    if args['do_eval']:
        tokenizer = tokenizer_class.from_pretrained(args['output_dir'], do_lower_case=args['do_lower_case'])
        model = model_class.from_pretrained(args['output_dir'])
        eval_batch_size = args['per_gpu_eval_batch_size'] * max(1, len(args['n_gpu']))
        eval_dataset, _ = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, eval_batch_size, mode="dev")
        y_true, y_pred = evaluate(args, model, labels, eval_dataset, pad_token_label_id)
        output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")

        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            report = metrics.classification_report(y_true, y_pred, digits=4)
            logging.info(report)
            writer.write(report)
    
    if args['do_predict']:
        tokenizer = tokenizer_class.from_pretrained(args['output_dir'], do_lower_case=args['do_lower_case'])
        model = model_class.from_pretrained(args['output_dir'])
        eval_batch_size = args['per_gpu_eval_batch_size'] * max(1, len(args['n_gpu']))
        predict_dataset, _ = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, eval_batch_size, mode="test")
        y_true, y_pred = evaluate(args, model, labels, predict_dataset, pad_token_label_id)
        output_test_predictions_file = os.path.join(args['output_dir'], "test_predictions.txt")

        with tf.io.gfile.GFile(output_test_predictions_file, "w") as writer:
            with tf.io.gfile.GFile(os.path.join(args['data_dir'], "test.txt"), "r") as f:
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
    flags.mark_flag_as_required("model_type")
    app.run(main)