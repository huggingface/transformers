import logging
from argparse import ArgumentParser

from transformers.commands import BaseTransformersCLICommand
from transformers.pipelines import SUPPORTED_TASKS, Pipeline, PipelineDataFormat, pipeline


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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


def run_command_factory(args):
    nlp = pipeline(
        task=args.task,
        model=args.model if args.model else None,
        config=args.config,
        tokenizer=args.tokenizer,
        device=args.device,
    )
    format = try_infer_format_from_ext(args.input) if args.format == "infer" else args.format
    reader = PipelineDataFormat.from_str(
        format=format,
        output_path=args.output,
        input_path=args.input,
        column=args.column if args.column else nlp.default_input_names,
        overwrite=args.overwrite,
    )
    return RunCommand(nlp, reader)


class RunCommand(BaseTransformersCLICommand):
    def __init__(self, nlp: Pipeline, reader: PipelineDataFormat):
        self._nlp = nlp
        self._reader = reader

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_parser = parser.add_parser("run", help="Run a pipeline through the CLI")
        run_parser.add_argument("--task", choices=SUPPORTED_TASKS.keys(), help="Task to run")
        run_parser.add_argument("--input", type=str, help="Path to the file to use for inference")
        run_parser.add_argument("--output", type=str, help="Path to the file that will be used post to write results.")
        run_parser.add_argument("--model", type=str, help="Name or path to the model to instantiate.")
        run_parser.add_argument("--config", type=str, help="Name or path to the model's config to instantiate.")
        run_parser.add_argument(
            "--tokenizer", type=str, help="Name of the tokenizer to use. (default: same as the model name)"
        )
        run_parser.add_argument(
            "--column",
            type=str,
            help="Name of the column to use as input. (For multi columns input as QA use column1,columns2)",
        )
        run_parser.add_argument(
            "--format",
            type=str,
            default="infer",
            choices=PipelineDataFormat.SUPPORTED_FORMATS,
            help="Input format to read from",
        )
        run_parser.add_argument(
            "--device",
            type=int,
            default=-1,
            help="Indicate the device to run onto, -1 indicates CPU, >= 0 indicates GPU (default: -1)",
        )
        run_parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file.")
        run_parser.set_defaults(func=run_command_factory)

    def run(self):
        nlp, outputs = self._nlp, []

        for entry in self._reader:
            output = nlp(**entry) if self._reader.is_multi_columns else nlp(entry)
            if isinstance(output, dict):
                outputs.append(output)
            else:
                outputs += output

        # Saving data
        if self._nlp.binary_output:
            binary_path = self._reader.save_binary(outputs)
            logger.warning("Current pipeline requires output to be in binary format, saving at {}".format(binary_path))
        else:
            self._reader.save(outputs)
