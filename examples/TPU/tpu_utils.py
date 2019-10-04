import tensorflow as tf


def get_tpu():
    tpu = None

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    except ValueError as e:
        print(e)

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://192.168.32.2:8470")
    except ValueError as e:
        print(e)

    # Select appropriate distribution strategy
    if tpu:
        # TF 2.0 change here: experimental_connect_to_cluster and initialize_tpu_system are now necessary
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        # TF 2.0 change here: steps_per_run does not exist anymore and is not needed
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    else:
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
        print('Running on CPU or GPU')
    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    return strategy, strategy.num_replicas_in_sync
