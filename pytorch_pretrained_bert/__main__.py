# coding: utf8
def main():
    import sys
    if (len(sys.argv) != 4 and len(sys.argv) != 5) or sys.argv[1] not in [
        "convert_tf_checkpoint_to_pytorch",
        "convert_openai_checkpoint",
        "convert_transfo_xl_checkpoint",
        "convert_gpt2_checkpoint",
    ]:
        print(
        "Should be used as one of: \n"
        ">> `pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT`, \n"
        ">> `pytorch_pretrained_bert convert_openai_checkpoint OPENAI_GPT_CHECKPOINT_FOLDER_PATH PYTORCH_DUMP_OUTPUT [OPENAI_GPT_CONFIG]`, \n"
        ">> `pytorch_pretrained_bert convert_transfo_xl_checkpoint TF_CHECKPOINT_OR_DATASET PYTORCH_DUMP_OUTPUT [TF_CONFIG]` or \n"
        ">> `pytorch_pretrained_bert convert_gpt2_checkpoint TF_CHECKPOINT PYTORCH_DUMP_OUTPUT [GPT2_CONFIG]`")
    else:
        if sys.argv[1] == "convert_tf_checkpoint_to_pytorch":
            try:
                from .convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
            except ImportError:
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
        elif sys.argv[1] == "convert_openai_checkpoint":
            from .convert_openai_checkpoint_to_pytorch import convert_openai_checkpoint_to_pytorch
            OPENAI_GPT_CHECKPOINT_FOLDER_PATH = sys.argv[2]
            PYTORCH_DUMP_OUTPUT = sys.argv[3]
            if len(sys.argv) == 5:
                OPENAI_GPT_CONFIG = sys.argv[4]
            else:
                OPENAI_GPT_CONFIG = ""
            convert_openai_checkpoint_to_pytorch(OPENAI_GPT_CHECKPOINT_FOLDER_PATH,
                                                 OPENAI_GPT_CONFIG,
                                                 PYTORCH_DUMP_OUTPUT)
        elif sys.argv[1] == "convert_transfo_xl_checkpoint":
            try:
                from .convert_transfo_xl_checkpoint_to_pytorch import convert_transfo_xl_checkpoint_to_pytorch
            except ImportError:
                print("pytorch_pretrained_bert can only be used from the commandline to convert TensorFlow models in PyTorch, "
                    "In that case, it requires TensorFlow to be installed. Please see "
                    "https://www.tensorflow.org/install/ for installation instructions.")
                raise

            if 'ckpt' in sys.argv[2].lower():
                TF_CHECKPOINT = sys.argv[2]
                TF_DATASET_FILE = ""
            else:
                TF_DATASET_FILE = sys.argv[2]
                TF_CHECKPOINT = ""
            PYTORCH_DUMP_OUTPUT = sys.argv[3]
            if len(sys.argv) == 5:
                TF_CONFIG = sys.argv[4]
            else:
                TF_CONFIG = ""
            convert_transfo_xl_checkpoint_to_pytorch(TF_CHECKPOINT, TF_CONFIG, PYTORCH_DUMP_OUTPUT, TF_DATASET_FILE)
        else:
            try:
                from .convert_gpt2_checkpoint_to_pytorch import convert_gpt2_checkpoint_to_pytorch
            except ImportError:
                print("pytorch_pretrained_bert can only be used from the commandline to convert TensorFlow models in PyTorch, "
                    "In that case, it requires TensorFlow to be installed. Please see "
                    "https://www.tensorflow.org/install/ for installation instructions.")
                raise

            TF_CHECKPOINT = sys.argv[2]
            PYTORCH_DUMP_OUTPUT = sys.argv[3]
            if len(sys.argv) == 5:
                TF_CONFIG = sys.argv[4]
            else:
                TF_CONFIG = ""
            convert_gpt2_checkpoint_to_pytorch(TF_CHECKPOINT, TF_CONFIG, PYTORCH_DUMP_OUTPUT)
if __name__ == '__main__':
    main()
