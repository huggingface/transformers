# coding: utf8
def main():
    import sys
    try:
        from .convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
    except ModuleNotFoundError:
        print("pytorch_pretrained_bert can only be used from the commandline to convert TensorFlow models in PyTorch, "
              "In that case, it requires TensorFlow to be installed. Please see "
              "https://www.tensorflow.org/install/ for installation instructions.")
        raise

    if len(sys.argv) != 5:
        # pylint: disable=line-too-long
        print("Should be used as `pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT`")
    else:
        PYTORCH_DUMP_OUTPUT = sys.argv.pop()
        TF_CONFIG = sys.argv.pop()
        TF_CHECKPOINT = sys.argv.pop()
        convert_tf_checkpoint_to_pytorch(TF_CHECKPOINT, TF_CONFIG, PYTORCH_DUMP_OUTPUT)

if __name__ == '__main__':
    main()
