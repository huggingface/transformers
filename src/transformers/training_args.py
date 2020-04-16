from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Overwrite the content of the output directory"}
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    evaluate_during_training: bool = field(
        default=False, metadata={"help": "Run evaluation during training at each logging step."}
    )

    per_gpu_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for training."})
    per_gpu_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": "Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default"
        },
    )
    eval_all_checkpoints: bool = field(
        default=False,
        metadata={
            "help": "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Avoid using CUDA even if it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html"
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
