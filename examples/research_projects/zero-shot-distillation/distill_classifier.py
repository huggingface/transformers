from dataclasses import dataclass, field, asdict
from typing import Optional, List
from tqdm.auto import tqdm
import sys
import logging

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, HfArgumentParser
from datasets import Dataset
import torch


DESCRIPTION = """\
Distills an NLI-based zero-shot classifier to a smaller, more efficient model with a fixed set of candidate class
names. Useful for speeding up zero-shot classification in cases where labeled training data is not available, but
when only a single fixed set of classes is needed. Takes a teacher NLI model, student classifier model, unlabeled
dataset, and set of K possible class names. Yields a single classifier with K outputs corresponding to the provided
class names.
"""


@dataclass
class TeacherModelArguments:
    teacher_name_or_path: Optional[str] = field(
        default="roberta-large-mnli",
        metadata={"help": "The NLI/zero-shot teacher model to be distilled."}
    )
    hypothesis_template: Optional[str] = field(
        default="This example is {}.",
        metadata={
            "help": (
                "Template used to turn class names into mock hypotheses for teacher NLI model. Must include {{}}"
                "where class name is inserted."
            )
        }
    )
    teacher_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size for generating teacher predictions."}
    )
    multi_class: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Allow multiple classes to be true rather than forcing them to sum to 1 (sometimes called"
                "multi-class multi-label classification)."
            )
        }
    )
    temperature: Optional[float] = field(
        default=1.,
        metadata={"help": "Temperature applied to teacher softmax for distillation."}
    )


@dataclass
class StudentModelArguments:
    student_name_or_path: Optional[str] = field(
        default="distilbert-base-uncased",
        metadata={"help": "The NLI/zero-shot teacher model to be distilled."}
    )


@dataclass
class DataTrainingArguments:
    data_file: str = field(metadata={"help": "Text file with one unlabeled instance per line."})
    class_names_file: str = field(metadata={"help": "Text file with one class name per line."})


@dataclass
class DistillTrainingArguments(TrainingArguments):
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    num_train_epochs: float = field(default=1.0, metadata={"help": "Total number of training epochs to perform."})
    do_train: bool = field(default=True, metadata={"help": "Whether to run training of student model."})
    do_eval: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to evaluate the agreement of the final student predictions and the teacher predictions"
                "after training."
            )
        }
    )


class DistillationTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        target_p = inputs["labels"]
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs[0]

        loss = -torch.sum(target_p * logits.log_softmax(dim=-1), axis=-1).mean()

        if return_outputs:
            return loss, outputs

        return loss


def read_lines(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                lines.append(line)
    return lines

def get_premise_hypothesis_pairs(examples, class_names, hypothesis_template):
    premises = []
    hypotheses = []
    for example in examples:
        for name in class_names:
            premises.append(example)
            hypotheses.append(hypothesis_template.format(name))
    return premises, hypotheses

def get_entailment_id(config):
    for label, ind in config.label2id.items():
        if label.lower().startswith("entail"):
            return ind
    logging.warning("Could not identify entailment dimension from teacher config label2id. Setting to -1.")
    return -1

def get_teacher_predictions(
        model_path: str,
        examples: List[str],
        class_names: List[str],
        hypothesis_template: str,
        batch_size: int,
        temperature: float,
        multi_class: bool,
        device: torch.device
    ):
    """
    Gets teachers by the same method as the zero-shot pipeline, but with more efficient mini-batching than pipelines
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    premises, hypotheses = get_premise_hypothesis_pairs(examples, class_names, hypothesis_template)
    logits = []

    for i in tqdm(range(0, len(premises), batch_size)):
        batch_premises = premises[i:i+batch_size]
        batch_hypotheses = hypotheses[i:i+batch_size]

        encodings = tokenizer(
            batch_premises,
            batch_hypotheses,
            padding=True,
            truncation="only_first",
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encodings.to(device))
            logits.append(outputs.logits.detach().cpu())

    entail_id = get_entailment_id(model.config)
    contr_id = -1 if entail_id == 0 else 0
    logits = torch.cat(logits, dim=0) # N*K x 3
    nli_logits = logits.reshape(len(examples), len(class_names), -1)[..., [contr_id, entail_id]] # N x K x 2

    if not multi_class:
        nli_logp = nli_logits.log_softmax(-1) / temperature
        nli_prob = nli_logp.softmax(1)
    else:
        nli_prob = (nli_logits / temperature).softmax(-1)
    
    return nli_prob[..., 1] # N x K

def main():
    # 0. parse arguments
    parser = HfArgumentParser(
        (DataTrainingArguments, TeacherModelArguments, StudentModelArguments, DistillTrainingArguments),
        description=DESCRIPTION,
    )
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, teacher_args, student_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, teacher_args, student_args, training_args = parser.parse_args_into_dataclasses()

    # 1. read in data
    examples = read_lines(data_args.data_file)
    class_names = read_lines(data_args.class_names_file)

    # 2. get teacher predictions and load into dataset
    logging.info("Generating predictions from zero-shot teacher model")
    teacher_device = torch.device("cuda:0") if torch.cuda.is_available() and not training_args.no_cuda else torch.device("cpu")
    teacher_soft_preds = get_teacher_predictions(
        teacher_args.teacher_name_or_path,
        examples,
        class_names,
        teacher_args.hypothesis_template,
        teacher_args.teacher_batch_size,
        teacher_args.temperature,
        teacher_args.multi_class,
        teacher_device,
    )
    dataset = Dataset.from_dict({
        "text": examples,
        "labels": teacher_soft_preds,
    })

    # 3. create student
    model = AutoModelForSequenceClassification.from_pretrained(student_args.student_name_or_path, num_labels=len(class_names))
    tokenizer = AutoTokenizer.from_pretrained(student_args.student_name_or_path, use_fast=True)
    model.config.id2label = {i: label for i, label in enumerate(class_names)}
    model.config.label2id = {label: i for i, label in enumerate(class_names)}

    # 4. train student on teacher predictions
    dataset = dataset.map(lambda ex: tokenizer(ex["text"]))
    dataset.set_format("torch")

    def compute_metrics(p, return_outputs=False):
        preds = p.predictions.argmax(-1)
        proxy_labels = p.label_ids.argmax(-1)
        return {"agreement": (preds == proxy_labels).mean().item()}

    trainer = DistillationTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        logging.info("Training student model on teacher predictions")
        trainer.train()

    if training_args.do_eval:
        agreement = trainer.evaluate(eval_dataset=dataset)['eval_agreement']
        logging.info(f"Agreement of student and teacher predictions: {agreement * 100:0.2f}%")

    trainer.save_model()


if __name__ == "__main__":
    main()
