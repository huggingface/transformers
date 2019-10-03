import tensorflow as tf
import tensorflow_datasets
from transformers import glue_convert_examples_to_features, glue_processors


def decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


def file_based_input_fn_builder(input_file, name_to_features):
    """Creates an `input_fn` closure to be passed for BERT custom training."""

    def input_fn():
        """Returns dataset for training/evaluation."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        d = d.map(lambda record: decode_record(record, name_to_features))

        # When `input_file` is a path to a single file or a list
        # containing a single path, disable auto sharding so that
        # same input file is sent to all workers.
        if isinstance(input_file, str) or len(input_file) == 1:
            options = tf.data.Options()
            options.experimental_distribute.auto_shard = False
            d = d.with_options(options)
        return d

    return input_fn


def create_classifier_dataset(tokenizer,
                              file_path,
                              seq_length,
                              batch_size,
                              is_training=True,
                              drop_remainder=True):
    """Creates input dataset from (tf)records files for train/eval."""
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'label_ids': tf.io.FixedLenFeature([], tf.int64),
        'is_real_example': tf.io.FixedLenFeature([], tf.int64),
    }
    input_fn = file_based_input_fn_builder(file_path, name_to_features)
    dataset = input_fn()

    def _select_data_from_record(record):
        x = {
            'input_word_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'input_type_ids': record['segment_ids']
        }
        y = record['label_ids']
        return (x, y)

    dataset = dataset.map(_select_data_from_record)

    # if is_training:
    #   dataset = dataset.shuffle(100)
    #   dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(1024)
    return dataset


def create_dataset(tokenizer,
                   file_path,
                   seq_length,
                   batch_size,
                   is_training=True,
                   drop_remainder=False
):
    processor = glue_processors["mnli"]()
    examples = processor.get_dev_examples(file_path)
    features = glue_convert_examples_to_features(examples, tokenizer, seq_length, 'mnli')

    all_input_ids = tf.constant([f.input_ids for f in features])
    all_attention_masks = tf.constant([f.attention_mask for f in features])
    all_token_type_ids = tf.constant([f.token_type_ids for f in features])
    all_labels = tf.constant([f.label for f in features])

    dataset = tf.data.Dataset.from_tensor_slices(({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "token_type_ids": all_token_type_ids
    }, all_labels))

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(1024)

    return dataset


if __name__ == "__main__":
    from transformers import BertTokenizer

    train_data_path = "/home/lysandre/transformers/examples/TPU/glue_data/MNLI"
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    seq_length = 128
    batch_size = 32

    create_dataset(
        tokenizer, train_data_path, seq_length, batch_size
    )
