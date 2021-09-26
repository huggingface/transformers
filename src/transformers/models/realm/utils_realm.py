import tensorflow.compat.v1 as tf

def load_scann_searcher(db,
                        num_neighbors,
                        dimensions_per_block=2,
                        num_leaves=1000,
                        num_leaves_to_search=100,
                        training_sample_size=100000):
    """Load scann searcher from checkpoint."""
    
    from scann.scann_ops.py.scann_ops_pybind import builder as Builder
        

    builder = Builder(
        db=db,
        num_neighbors=num_neighbors,
        distance_measure="dot_product")
    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=num_leaves_to_search,
        training_sample_size=training_sample_size)
    builder = builder.score_ah(dimensions_per_block=dimensions_per_block)

    searcher = builder.build()
    return searcher

def convert_tfrecord_to_np(block_records_path, num_block_records):
    blocks_dataset = tf.data.TFRecordDataset(
        block_records_path, buffer_size=512 * 1024 * 1024)
    blocks_dataset = blocks_dataset.batch(
        num_block_records, drop_remainder=True)
    np_record = [raw_record.numpy() for raw_record in blocks_dataset.take(1)][0]

    return np_record