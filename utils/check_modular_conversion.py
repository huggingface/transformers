import argparse
import difflib
import glob
import logging
import subprocess
from io import StringIO

from create_dependency_mapping import find_priority_list

# Console for rich printing
from modular_model_converter import convert_modular_file
from rich.console import Console
from rich.syntax import Syntax


logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
console = Console()


def process_file(modular_file_path, generated_modeling_content, file_type="modeling_", fix_and_overwrite=False):
    file_name_prefix = file_type.split("*")[0]
    file_name_suffix = file_type.split("*")[-1] if "*" in file_type else ""
    file_path = modular_file_path.replace("modular_", f"{file_name_prefix}_").replace(".py", f"{file_name_suffix}.py")
    # Read the actual modeling file
    with open(file_path, "r") as modeling_file:
        content = modeling_file.read()
    output_buffer = StringIO(generated_modeling_content[file_type][0])
    output_buffer.seek(0)
    output_content = output_buffer.read()
    diff = difflib.unified_diff(
        output_content.splitlines(),
        content.splitlines(),
        fromfile=f"{file_path}_generated",
        tofile=f"{file_path}",
        lineterm="",
    )
    diff_list = list(diff)
    # Check for differences
    if diff_list:
        if fix_and_overwrite:
            with open(file_path, "w") as modeling_file:
                modeling_file.write(generated_modeling_content[file_type][0])
            console.print(f"[bold blue]Overwritten {file_path} with the generated content.[/bold blue]")
        else:
            console.print(f"\n[bold red]Differences found between the generated code and {file_path}:[/bold red]\n")
            diff_text = "\n".join(diff_list)
            syntax = Syntax(diff_text, "diff", theme="ansi_dark", line_numbers=True)
            console.print(syntax)
        return 1
    else:
        console.print(f"[bold green]No differences found for {file_path}.[/bold green]")
        return 0


def compare_files(modular_file_path, fix_and_overwrite=False):
    # Generate the expected modeling content
    generated_modeling_content = convert_modular_file(modular_file_path)
    diff = 0
    for file_type in generated_modeling_content.keys():
        diff += process_file(modular_file_path, generated_modeling_content, file_type, fix_and_overwrite)
    return diff


def get_models_in_diff():
    """
    Finds all models that have been modified in the diff.

    Returns:
        A set containing the names of the models that have been modified (e.g. {'llama', 'whisper'}).
    """
    fork_point_sha = subprocess.check_output("git merge-base add-fast HEAD".split()).decode("utf-8")
    modified_files = (
        subprocess.check_output(f"git diff --diff-filter=d --name-only {fork_point_sha}".split())
        .decode("utf-8")
        .split()
    )

    # Matches both modelling files and tests
    relevant_modified_files = [x for x in modified_files if "/models/" in x and x.endswith(".py")]
    model_names = set()
    for file_path in relevant_modified_files:
        model_name = file_path.split("/")[-2]
        model_names.add(model_name)
    return model_names


def guaranteed_no_diff(modular_file_path, dependencies, models_in_diff):
    """
    Returns whether it is guaranteed to have no differences between the modular file and the modeling file.

    Model is in the diff -> not guaranteed to have no differences
    Dependency is in the diff -> not guaranteed to have no differences
    Otherwise -> guaranteed to have no differences

    Args:
        modular_file_path: The path to the modular file.
        dependencies: A dictionary containing the dependencies of each modular file.
        models_in_diff: A set containing the names of the models that have been modified.

    Returns:
        A boolean indicating whether the model (code and tests) is guaranteed to have no differences.
    """
    model_name = modular_file_path.rsplit("modular_", 1)[1].replace(".py", "")
    if model_name in models_in_diff:
        return False
    for dep in dependencies[modular_file_path]:
        # two possible patterns: `transformers.models.model_name.(...)` or `model_name.(...)`
        dependency_model_name = dep.split(".")[-2]
        if dependency_model_name in models_in_diff:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare modular_xxx.py files with modeling_xxx.py files.")
    parser.add_argument(
        "--files", default=["all"], type=list, nargs="+", help="List of modular_xxx.py files to compare."
    )
    parser.add_argument(
        "--fix_and_overwrite", action="store_true", help="Overwrite the modeling_xxx.py file if differences are found."
    )
    args = parser.parse_args()
    if args.files == ["all"]:
        args.files = glob.glob("src/transformers/models/**/modular_*.py", recursive=True)

    # Assuming there is a topological sort on the dependency mapping: if the file being checked and its dependencies
    # are not in the diff, then there it is guaranteed to have no differences. If no models are in the diff, then this
    # script will do nothing.
    models_in_diff = get_models_in_diff()
    if not models_in_diff:
        console.print("[bold green]No models files or model tests in the diff, skipping modular checks[/bold green]")
        exit(0)

    skipped_models = set()
    non_matching_files = 0
    ordered_files, dependencies = find_priority_list(args.files)
    for modular_file_path in ordered_files:
        is_guaranteed_no_diff = guaranteed_no_diff(modular_file_path, dependencies, models_in_diff)
        if is_guaranteed_no_diff:
            model_name = modular_file_path.rsplit("modular_", 1)[1].replace(".py", "")
            skipped_models.add(model_name)
            continue
        non_matching_files += compare_files(modular_file_path, args.fix_and_overwrite)
        models_in_diff = get_models_in_diff()  # When overwriting, the diff changes

    if non_matching_files and not args.fix_and_overwrite:
        raise ValueError("Some diff and their modeling code did not match.")

    if skipped_models:
        console.print(
            f"[bold green]Skipped {len(skipped_models)} models and their dependencies that are not in the diff: "
            f"{', '.join(skipped_models)}[/bold green]"
        )
