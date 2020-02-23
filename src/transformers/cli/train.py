import os
from logging import getLogger
import typer

from transformers import SingleSentenceClassificationProcessor as Processor
from transformers import TextClassificationPipeline, is_tf_available, is_torch_available


if not is_tf_available() and not is_torch_available():
    raise RuntimeError("At least one of PyTorch or TensorFlow 2.0+ should be installed to use CLI training")

# TF training parameters
USE_XLA = False
USE_AMP = False


def train(
    task: str = typer.Argument(...),
    model: str = typer.Option("bert-base-uncased", help="Name or path to the model to instantiate."),
    train_data: str = typer.Option(
        ...,
        help="path to train (and optionally evaluation) dataset as a csv with " "tab separated labels and sentences.",
    ),
    validation_data: str = typer.Option(None, help="path to validation dataset."),
    validation_split: float = typer.Option(
        0.1, help="if validation dataset is not provided, fraction of train dataset " "to use as validation dataset."
    ),
    output: str = typer.Option(..., help="Path to the file that will be used post to write results."),
    column_label: int = typer.Option(0, help="Column of the dataset csv file with example labels."),
    column_text: int = typer.Option(0, help="Column of the dataset csv file with example texts."),
    column_id: int = typer.Option(0, help="Column of the dataset csv file with example ids."),
    skip_first_row: bool = False,
    train_batch_size: int = 32,
    valid_batch_size: int = 64,
    learning_rate: float = 3e-5,
    adam_epsilon: float = 1e-08,
):

    """Train a new model on TASK."""

    logger = getLogger("transformers-cli/training")
    os.makedirs(output, exist_ok=True)

    logger.info("Loading {} pipeline for {}".format(task, model))

    if task == "text_classification":
        pipeline = TextClassificationPipeline.from_pretrained(model)
    elif task == "token_classification":
        raise NotImplementedError
    elif task == "question_answering":
        raise NotImplementedError

    logger.info("Loading dataset from {}".format(train_data))

    train_dataset = Processor.create_from_csv(
        train_data,
        column_label=column_label,
        column_text=column_text,
        column_id=column_id,
        skip_first_row=skip_first_row,
    )
    valid_dataset = None

    if validation_data:
        logger.info("Loading validation dataset from {}".format(validation_data))
        valid_dataset = Processor.create_from_csv(
            validation_data,
            column_label=column_label,
            column_text=column_text,
            column_id=column_id,
            skip_first_row=skip_first_row,
        )

    if is_tf_available():
        pipeline.fit(
            train_dataset,
            validation_data=valid_dataset,
            validation_split=validation_split,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
        )

        # Save trained pipeline
        pipeline.save_pretrained(output)
    else:
        raise NotImplementedError
