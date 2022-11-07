from dataclasses import dataclass, field
from typing import Optional

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default="prompt", metadata={"help": "Few-shot learning model type. Choice: finetune, prompt"}
    )
    demo: bool = field(default=False, metadata={"help": "Whether or not to use demonstration."})
    model_type: str = field(default="roberta")
    knn_lambda: float = field(default=0.2,)
    knn_topk: Optional[int] = field(default=32, metadata={"help": "knn topk"})
    beta: float = field(default=0.2,)


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """

    num_k: Optional[int] = field(default=16, metadata={"help": "Number of training instances per class"})

    # For prompting
    template: str = field(default=None, metadata={"help": "Template"})

    demo_template: str = field(default=None)

    mapping: str = field(default="{'0':'terrible','1':'great'}", metadata={"help": "Label word mapping"})

    # For logging
    tag: str = field(default="", metadata={"help": "Set the tag and find the result easier in the log."})

    # For max length
    first_sent_limit: int = field(
        default=None, metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )
    other_sent_limit: int = field(
        default=None, metadata={"help": "Limit the length of sentences other than the first sentence"}
    )
    demo_max_length: int = field(default=None, metadata={"help": "the maximum length of sentence in demonstration"})
    demo_first_sent_limit: int = field(default=None, metadata={"help": "Limit the length of the first sentence."})
    demo_other_sent_limit: int = field(default=None, metadata={"help": "Limit the length of the other sentence."})
    virtual_demo: bool = field(default=False)
    virtual_demo_length_per_label: int = field(default=0,)
    virtual_demo_init: str = field(default="random",)

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(default=False, metadata={"help": "Whether to use prompt-based fine-tuning"})
    demo_num: int = field(default=1,)
    demo_topk: int = field(default=16,)


@dataclass
class DynamicTrainingArguments(TrainingArguments):

    # For ensemble
    array_id: int = field(
        default=-1, metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1, metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False, metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(default=None, metadata={"help": "Where to save the prediction result"})

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"},
    )

    # Contrastive learning
    lambda_cl: float = field(default=0.01,)

    # Turn off train/test
    no_train: bool = field(default=False, metadata={"help": "No training"})
    no_predict: bool = field(default=False, metadata={"help": "No test"})
    evaluate_during_training: bool = field(default=True, metadata={"help": "evaluate during training."})
    knn_mode: bool = field(default=False)
    ckpt_dir: str = field(default=None)
    use_source_datastore: bool = field(default=False,)
    train_with_knn: bool = field(default=False,)
    only_train_knn: bool = field(default=False)
    compute_mem: bool = field(default=False)
    use_demo: bool = field(default=False)
    START: int = field(default=0,)
    LENGTH: int = field(default=0,)
    do_case: bool = field(default=False)
