from logging import getLogger
from pathlib import Path

import typer

from transformers.cli._types import ModelType


def convert(
    model_type: ModelType,
    tf_checkpoint: Path,
    pytorch_dump_output: Path,
    config: str = typer.Option("", help="Configuration file path or folder."),
    finetuning_task_name: str = typer.Option(
        None, help="Optional fine-tuning task name if the TF model was a finetuned model."
    ),
):
    """Convert model from original author checkpoints 
    to Transformers PyTorch checkpoints.

    Usage:
    ```bash
    $ transformers convert bert ./tf_checkpoint ./output_dir
    ```
    """

    logger = getLogger("transformers-cli/converting")
    logger.info("Loading model {}".format(model_type))

    if model_type == ModelType.BERT:
        try:
            from transformers.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
        except ImportError:
            msg = (
                "transformers can only be used from the commandline to convert TensorFlow models in PyTorch, "
                "In that case, it requires TensorFlow to be installed. Please see "
                "https://www.tensorflow.org/install/ for installation instructions."
            )
            raise ImportError(msg)

        convert_tf_checkpoint_to_pytorch(tf_checkpoint, config, pytorch_dump_output)
    elif model_type == ModelType.GPT:
        from transformers.convert_openai_original_tf_checkpoint_to_pytorch import convert_openai_checkpoint_to_pytorch

        convert_openai_checkpoint_to_pytorch(tf_checkpoint, config, pytorch_dump_output)
    elif model_type == ModelType.TRANSFORMER_XL:
        try:
            from transformers.convert_transfo_xl_original_tf_checkpoint_to_pytorch import (
                convert_transfo_xl_checkpoint_to_pytorch,
            )
        except ImportError:
            msg = (
                "transformers can only be used from the commandline to convert TensorFlow models in PyTorch, "
                "In that case, it requires TensorFlow to be installed. Please see "
                "https://www.tensorflow.org/install/ for installation instructions."
            )
            raise ImportError(msg)

        if "ckpt" in tf_checkpoint.lower():
            TF_CHECKPOINT = tf_checkpoint
            TF_DATASET_FILE = ""
        else:
            TF_DATASET_FILE = tf_checkpoint
            TF_CHECKPOINT = ""
        convert_transfo_xl_checkpoint_to_pytorch(TF_CHECKPOINT, config, pytorch_dump_output, TF_DATASET_FILE)
    elif model_type == ModelType.GPT2:
        try:
            from transformers.convert_gpt2_original_tf_checkpoint_to_pytorch import convert_gpt2_checkpoint_to_pytorch
        except ImportError:
            msg = (
                "transformers can only be used from the commandline to convert TensorFlow models in PyTorch, "
                "In that case, it requires TensorFlow to be installed. Please see "
                "https://www.tensorflow.org/install/ for installation instructions."
            )
            raise ImportError(msg)

        convert_gpt2_checkpoint_to_pytorch(tf_checkpoint, config, pytorch_dump_output)
    elif model_type == ModelType.XLNET:
        try:
            from transformers.convert_xlnet_original_tf_checkpoint_to_pytorch import (
                convert_xlnet_checkpoint_to_pytorch,
            )
        except ImportError:
            msg = (
                "transformers can only be used from the commandline to convert TensorFlow models in PyTorch, "
                "In that case, it requires TensorFlow to be installed. Please see "
                "https://www.tensorflow.org/install/ for installation instructions."
            )
            raise ImportError(msg)

        convert_xlnet_checkpoint_to_pytorch(tf_checkpoint, config, pytorch_dump_output, finetuning_task_name)
    elif model_type == ModelType.XLM:
        from transformers.convert_xlm_original_pytorch_checkpoint_to_pytorch import convert_xlm_checkpoint_to_pytorch

        convert_xlm_checkpoint_to_pytorch(tf_checkpoint, pytorch_dump_output)
