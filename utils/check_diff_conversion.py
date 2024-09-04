import argparse
import difflib
import glob
import logging
from io import StringIO

# Console for rich printing
from diff_model_converter import convert_diff_file
from rich.console import Console
from rich.syntax import Syntax


logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
console = Console()


def compare_files(diff_file_path, fix_and_overwrite=False):
    # Generate the expected modeling content
    generated_modeling_content = convert_diff_file(diff_file_path)
    modeling_file_path = diff_file_path.replace("diff_", "modeling_")
    # Read the actual modeling file
    with open(modeling_file_path, "r") as modeling_file:
        modeling_content = modeling_file.read()

    # Store the output in a buffer
    output_buffer = StringIO(generated_modeling_content[1])

    # Read the buffer content
    output_buffer.seek(0)
    output_content = output_buffer.read()

    # Compare input and output contents
    diff = difflib.unified_diff(
        modeling_content.splitlines(),
        output_content.splitlines(),
        fromfile=f"{modeling_file_path}_generated",
        tofile=f"{modeling_file_path}",
        lineterm="",
    )

    # Convert the diff generator to a list for further processing
    diff_list = list(diff)

    # Check for differences
    if diff_list:
        if fix_and_overwrite:
            with open(modeling_file_path, "w") as modeling_file:
                modeling_file.write(generated_modeling_content)
            console.print(f"[bold blue]Overwritten {modeling_file_path} with the generated content.[/bold blue]")
        else:
            console.print(
                f"\n[bold red]Differences found between the generated modeling code and {modeling_file_path}:[/bold red]\n"
            )
            diff_text = "\n".join(diff_list)
            syntax = Syntax(diff_text, "diff", theme="ansi_dark", line_numbers=True)
            console.print(syntax)
        return 1
    else:
        console.print(f"\n[bold green]No differences found for {modeling_file_path}.[/bold green]")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare diff_xxx.py files with modeling_xxx.py files.")
    parser.add_argument("--files", default=["all"], type=list, nargs="+", help="List of diff_xxx.py files to compare.")
    parser.add_argument(
        "--fix_and_overwrite", action="store_true", help="Overwrite the modeling_xxx.py file if differences are found."
    )
    args = parser.parse_args()
    if args.files == ["all"]:
        args.files = glob.glob("src/transformers/models/**/diff_*.py", recursive=True)
    non_matching_files = 0
    for diff_file_path in args.files:
        non_matching_files += compare_files(diff_file_path, args.fix_and_overwrite)

    if non_matching_files and not args.fix_and_overwrite:
        raise ValueError("Some diff and their modeling code did not match.")
