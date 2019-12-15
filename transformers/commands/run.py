from argparse import ArgumentParser

from transformers.commands import BaseTransformersCLICommand
from transformers.pipelines import pipeline, Pipeline, PipelineDataFormat, SUPPORTED_TASKS


def try_infer_format_from_ext(path: str):
    for ext in PipelineDataFormat.SUPPORTED_FORMATS:
        if path.endswith(ext):
            return ext

    raise Exception(
        'Unable to determine file format from file extension {}. '
        'Please provide the format through --format {}'.format(path, PipelineDataFormat.SUPPORTED_FORMATS)
    )


def run_command_factory(args):
    nlp = pipeline(task=args.task, model=args.model, tokenizer=args.tokenizer)
    format = try_infer_format_from_ext(args.input) if args.format == 'infer' else args.format
    reader = PipelineDataFormat.from_str(format, args.output, args.input, args.column)
    return RunCommand(nlp, reader)


class RunCommand(BaseTransformersCLICommand):

    def __init__(self, nlp: Pipeline, reader: PipelineDataFormat):
        self._nlp = nlp
        self._reader = reader

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_parser = parser.add_parser('run', help="Run a pipeline through the CLI")
        run_parser.add_argument('--task', choices=SUPPORTED_TASKS.keys(), help='Task to run')
        run_parser.add_argument('--model', type=str, required=True, help='Name or path to the model to instantiate.')
        run_parser.add_argument('--tokenizer', type=str, help='Name of the tokenizer to use. (default: same as the model name)')
        run_parser.add_argument('--column', type=str, required=True, help='Name of the column to use as input. (For multi columns input as QA use column1,columns2)')
        run_parser.add_argument('--format', type=str, default='infer', choices=PipelineDataFormat.SUPPORTED_FORMATS, help='Input format to read from')
        run_parser.add_argument('--input', type=str, required=True, help='Path to the file to use for inference')
        run_parser.add_argument('--output', type=str, required=True, help='Path to the file that will be used post to write results.')
        run_parser.add_argument('kwargs', nargs='*', help='Arguments to forward to the file format reader')
        run_parser.set_defaults(func=run_command_factory)

    def run(self):
        nlp, output = self._nlp, []
        for entry in self._reader:
            if self._reader.is_multi_columns:
                output += [nlp(**entry)]
            else:
                output += [nlp(entry)]

        # Saving data
        self._reader.save(output)



