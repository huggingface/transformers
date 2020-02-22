from enum import Enum
import logging
from pathlib import Path

import typer
from transformers.cli._types import SupportedFormat, SupportedTask
from transformers.pipelines import SUPPORTED_TASKS, PipelineDataFormat, pipeline


def run(
    task: SupportedTask,
    input_path: Path,
    output_path: Path,
    model: str = typer.Option(None, help="Name or path to the model to instantiate."),
    config: str = typer.Option(..., help="Name or path to the model's config to instantiate."),
    tokenizer: str = typer.Option(None, help="Name of the tokenizer to use. (default: same as the model name)"),
    column: str = typer.Option(..., help="Name of the column to use as input. (For multi columns input as QA use column1,columns2)"),
    data_format: SupportedFormat = typer.Option(SupportedFormat.INFER, help="Input format to read from"),
    device: int = typer.Option(-1, help="Indicate the device to run onto, -1 indicates CPU, >= 0 indicates GPU (default: -1)"),
    overwrite: bool = typer.Option(False, help="Allow overwriting the output file.")
):
    
    """CLI tool to run a pipeline TASK in on the data in INPUT_PATH. 
    Save the results to disk at OUTPUT_PATH"""

    logger = logging.getLogger(f"transformers-cli/run/{task.value}")
    
    nlp = pipeline(
        task=task.value,
        model=model if model else None,
        config=config,
        tokenizer=tokenizer,
        device=device
    )
    data_format = try_infer_format_from_ext(input_path) if data_format == SupportedFormat.INFER else data_format.value
    reader = PipelineDataFormat.from_str(
        format=data_format,
        output_path=output_path,
        input_path=input_path,
        column=column if column else nlp.default_input_names,
        overwrite=overwrite,
    )

    outputs = []

    for entry in reader:
        output = nlp(**entry) if reader.is_multi_columns else nlp(entry)
        if isinstance(output, dict):
            outputs.append(output)
        else:
            outputs += output

    # Saving data
    if nlp.binary_output:
        binary_path = reader.save_binary(outputs)
        logger.warning("Current pipeline requires output to be in binary format, saving at {}".format(binary_path))
    else:
        reader.save(outputs)


def try_infer_format_from_ext(path: str):
    if not path:
        return "pipe"

    for ext in PipelineDataFormat.SUPPORTED_FORMATS:
        if path.endswith(ext):
            return ext

    raise Exception(
        "Unable to determine file format from file extension {}. "
        "Please provide the format through --format {}".format(path, PipelineDataFormat.SUPPORTED_FORMATS)
    )
