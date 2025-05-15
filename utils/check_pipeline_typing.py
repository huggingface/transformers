import re

from transformers.pipelines import SUPPORTED_TASKS, Pipeline


HEADER = """
# fmt: off
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#                       The part of the file below was automatically generated from the code.
#           Do NOT edit this part of the file manually as any edits will be overwritten by the generation
#           of the file. If any change should be done, please apply the changes to the `pipeline` function
#            below and run `python utils/check_pipeline_typing.py --fix_and_overwrite` to update the file.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨

from typing import Literal, overload


"""

FOOTER = """
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#                       The part of the file above was automatically generated from the code.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# fmt: on
"""

TASK_PATTERN = "task: Optional[str] = None"


def main(pipeline_file_path: str, fix_and_overwrite: bool = False):
    with open(pipeline_file_path, "r") as file:
        content = file.read()

    # extract generated code in between <generated-code> and </generated-code>
    current_generated_code = re.search(r"# <generated-code>(.*)# </generated-code>", content, re.DOTALL).group(1)
    content_without_generated_code = content.replace(current_generated_code, "")

    # extract pipeline signature in between `def pipeline` and `-> Pipeline`
    pipeline_signature = re.search(r"def pipeline(.*) -> Pipeline:", content_without_generated_code, re.DOTALL).group(
        1
    )
    pipeline_signature = pipeline_signature.replace("(\n    ", "(")  # start of the signature
    pipeline_signature = pipeline_signature.replace(",\n    ", ", ")  # intermediate arguments
    pipeline_signature = pipeline_signature.replace(",\n)", ")")  # end of the signature

    # collect and sort available pipelines
    pipelines = [(f'"{task}"', task_info["impl"]) for task, task_info in SUPPORTED_TASKS.items()]
    pipelines = sorted(pipelines, key=lambda x: x[0])
    pipelines.insert(0, (None, Pipeline))

    # generate new `pipeline` signatures
    new_generated_code = ""
    for task, pipeline_class in pipelines:
        if TASK_PATTERN not in pipeline_signature:
            raise ValueError(f"Can't find `{TASK_PATTERN}` in pipeline signature: {pipeline_signature}")
        pipeline_type = pipeline_class if isinstance(pipeline_class, str) else pipeline_class.__name__
        new_pipeline_signature = pipeline_signature.replace(TASK_PATTERN, f"task: Literal[{task}]")
        new_generated_code += f"@overload\ndef pipeline{new_pipeline_signature} -> {pipeline_type}: ...\n"

    new_generated_code = HEADER + new_generated_code + FOOTER
    new_generated_code = new_generated_code.rstrip("\n") + "\n"

    if new_generated_code != current_generated_code and fix_and_overwrite:
        print(f"Updating {pipeline_file_path}...")
        wrapped_current_generated_code = "# <generated-code>" + current_generated_code + "# </generated-code>"
        wrapped_new_generated_code = "# <generated-code>" + new_generated_code + "# </generated-code>"
        content = content.replace(wrapped_current_generated_code, wrapped_new_generated_code)

        # write content to file
        with open(pipeline_file_path, "w") as file:
            file.write(content)

    elif new_generated_code != current_generated_code and not fix_and_overwrite:
        message = (
            f"Found inconsistencies in {pipeline_file_path}. "
            "Run `python utils/check_pipeline_typing.py --fix_and_overwrite` to fix them."
        )
        raise ValueError(message)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    parser.add_argument(
        "--pipeline_file_path",
        type=str,
        default="src/transformers/pipelines/__init__.py",
        help="Path to the pipeline file.",
    )
    args = parser.parse_args()
    main(args.pipeline_file_path, args.fix_and_overwrite)
