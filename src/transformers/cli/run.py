# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from enum import Enum
from typing import Annotated, Optional

import typer

from ..pipelines import PipelineDataFormat, get_supported_tasks, pipeline
from ..utils import logging


logger = logging.get_logger(__name__)

TaskEnum = Enum("TaskEnum", {task.upper(): task for task in get_supported_tasks()}, type=str)
FormatEnum = Enum("FormatEnum", {fmt.upper(): fmt for fmt in PipelineDataFormat.SUPPORTED_FORMATS}, type=str)


def run(
    task: Annotated[TaskEnum, typer.Argument(help="Task to run", case_sensitive=False)],  # type: ignore
    input: Annotated[Optional[str], typer.Option(help="Path to the file to use for inference")] = None,
    output: Annotated[
        Optional[str], typer.Option(help="Path to the file that will be used post to write results.")
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            help="Name or path to the model to instantiate. If not provided, will use the default model for that task."
        ),
    ] = None,
    config: Annotated[
        Optional[str],
        typer.Option(
            help="Name or path to the model's config to instantiate. If not provided, will use the model's one."
        ),
    ] = None,
    tokenizer: Annotated[
        Optional[str], typer.Option(help="Name of the tokenizer to use. If not provided, will use the model's one.")
    ] = None,
    column: Annotated[
        Optional[str],
        typer.Option(help="Name of the column to use as input. For multi columns input use 'column1,columns2'"),
    ] = None,
    format: Annotated[FormatEnum, typer.Option(help="Input format to read from", case_sensitive=False)] = "infer",  # type: ignore
    device: Annotated[
        int, typer.Option(help="Indicate the device to run onto, -1 indicates CPU, >= 0 indicates GPU.")
    ] = -1,
    overwrite: Annotated[bool, typer.Option(help="Allow overwriting the output file.")] = False,
):
    """Run a pipeline on a given input file."""
    # Initialize pipeline
    pipe = pipeline(task=task, model=model, config=config, tokenizer=tokenizer, device=device)

    # Initialize reader
    reader = PipelineDataFormat.from_str(
        format=_try_infer_format_from_ext(input) if format == "infer" else format,
        output_path=output,
        input_path=input,
        column=column if column else pipe.default_input_names,
        overwrite=overwrite,
    )

    # Run
    outputs = []
    for entry in reader:
        output = pipe(**entry) if reader.is_multi_columns else pipe(entry)
        if isinstance(output, dict):
            outputs.append(output)
        else:
            outputs += output

    # Saving data
    if pipe.binary_output:
        binary_path = reader.save_binary(outputs)
        logger.warning(f"Current pipeline requires output to be in binary format, saving at {binary_path}")
    else:
        reader.save(outputs)


def _try_infer_format_from_ext(path: str) -> str:
    if not path:
        return "pipe"

    for ext in PipelineDataFormat.SUPPORTED_FORMATS:
        if path.endswith(ext):
            return ext

    raise Exception(
        f"Unable to determine file format from file extension {path}. "
        f"Please provide the format through --format {PipelineDataFormat.SUPPORTED_FORMATS}"
    )
