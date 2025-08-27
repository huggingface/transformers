import argparse
import difflib
import glob
import logging
import multiprocessing
import os
import shutil
import subprocess
from functools import partial
from io import StringIO

from create_dependency_mapping import find_priority_list

# Console for rich printing
from modular_model_converter import convert_modular_file
from rich.console import Console
from rich.syntax import Syntax


logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
console = Console()

BACKUP_EXT = ".modular_backup"


def process_file(
    modular_file_path,
    generated_modeling_content,
    file_type="modeling_",
    show_diff=True,
):
    file_name_prefix = file_type.split("*")[0]
    file_name_suffix = file_type.split("*")[-1] if "*" in file_type else ""
    file_path = modular_file_path.replace("modular_", f"{file_name_prefix}_").replace(".py", f"{file_name_suffix}.py")
    # Read the actual modeling file
    with open(file_path, "r", encoding="utf-8") as modeling_file:
        content = modeling_file.read()
    output_buffer = StringIO(generated_modeling_content[file_type])
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
        # first save the copy of the original file, to be able to restore it later
        if os.path.exists(file_path):
            shutil.copy(file_path, file_path + BACKUP_EXT)
        # we always save the generated content, to be able to update dependant files
        with open(file_path, "w", encoding="utf-8", newline="\n") as modeling_file:
            modeling_file.write(generated_modeling_content[file_type])
        console.print(f"[bold blue]Overwritten {file_path} with the generated content.[/bold blue]")
        if show_diff:
            console.print(f"\n[bold red]Differences found between the generated code and {file_path}:[/bold red]\n")
            diff_text = "\n".join(diff_list)
            syntax = Syntax(diff_text, "diff", theme="ansi_dark", line_numbers=True)
            console.print(syntax)
        return 1
    else:
        console.print(f"[bold green]No differences found for {file_path}.[/bold green]")
        return 0


def compare_files(modular_file_path, show_diff=True):
    # Generate the expected modeling content
    generated_modeling_content = convert_modular_file(modular_file_path)
    diff = 0
    for file_type in generated_modeling_content:
        diff += process_file(modular_file_path, generated_modeling_content, file_type, show_diff)
    return diff


def get_models_in_diff():
    """
    Finds all models that have been modified in the diff.

    Returns:
        A set containing the names of the models that have been modified (e.g. {'llama', 'whisper'}).
    """
    fork_point_sha = subprocess.check_output("git merge-base main HEAD".split()).decode("utf-8")
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
        "--files", default=["all"], type=str, nargs="+", help="List of modular_xxx.py files to compare."
    )
    parser.add_argument(
        "--fix_and_overwrite", action="store_true", help="Overwrite the modeling_xxx.py file if differences are found."
    )
    parser.add_argument("--check_all", action="store_true", help="Check all files, not just the ones in the diff.")
    parser.add_argument(
        "--num_workers",
        default=-1,
        type=int,
        help="The number of workers to run. Default is -1, which means the number of CPU cores.",
    )
    args = parser.parse_args()
    if args.files == ["all"]:
        args.files = glob.glob("src/transformers/models/**/modular_*.py", recursive=True)

    if args.num_workers == -1:
        args.num_workers = multiprocessing.cpu_count()

    # Assuming there is a topological sort on the dependency mapping: if the file being checked and its dependencies
    # are not in the diff, then there it is guaranteed to have no differences. If no models are in the diff, then this
    # script will do nothing.
    current_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if current_branch == "main":
        console.print(
            "[bold red]You are developing on the main branch. We cannot identify the list of changed files and will have to check all files. This may take a while.[/bold red]"
        )
        models_in_diff = {file_path.split("/")[-2] for file_path in args.files}
    else:
        models_in_diff = get_models_in_diff()
        if not models_in_diff and not args.check_all:
            console.print(
                "[bold green]No models files or model tests in the diff, skipping modular checks[/bold green]"
            )
            exit(0)

    skipped_models = set()
    non_matching_files = []
    ordered_files, dependencies = find_priority_list(args.files)
    flat_ordered_files = [item for sublist in ordered_files for item in sublist]

    # ordered_files is a *sorted* list of lists of filepaths
    #  - files from the first list do NOT depend on other files
    #  - files in the second list depend on files from the first list
    #  - files in the third list depend on files from the second and (optionally) the first list
    #  - ... and so on
    # files (models) within the same list are *independent* of each other;
    # we start applying modular conversion to each list in parallel, starting from the first list

    console.print(f"[bold yellow]Number of dependency levels: {len(ordered_files)}[/bold yellow]")
    console.print(f"[bold yellow]Files per level: {tuple([len(x) for x in ordered_files])}[/bold yellow]")

    try:
        for dependency_level_files in ordered_files:
            # Filter files guaranteed no diff
            files_to_check = []
            for file_path in dependency_level_files:
                if not args.check_all and guaranteed_no_diff(file_path, dependencies, models_in_diff):
                    skipped_models.add(file_path.split("/")[-2])  # save model folder name
                else:
                    files_to_check.append(file_path)

            if not files_to_check:
                continue

            # Process files with diff
            num_workers = min(args.num_workers, len(files_to_check))
            with multiprocessing.Pool(num_workers) as p:
                is_changed_flags = p.map(
                    partial(compare_files, show_diff=not args.fix_and_overwrite),
                    files_to_check,
                )

            # Collect changed files and their original paths
            for is_changed, file_path in zip(is_changed_flags, files_to_check):
                if is_changed:
                    non_matching_files.append(file_path)

                    # Update changed models, after each round of conversions
                    # (save model folder name)
                    models_in_diff.add(file_path.split("/")[-2])

    finally:
        # Restore overwritten files by modular (if needed)
        backup_files = glob.glob("**/*" + BACKUP_EXT, recursive=True)
        for backup_file_path in backup_files:
            overwritten_path = backup_file_path.replace(BACKUP_EXT, "")
            if not args.fix_and_overwrite and os.path.exists(overwritten_path):
                shutil.copy(backup_file_path, overwritten_path)
            os.remove(backup_file_path)

    if non_matching_files and not args.fix_and_overwrite:
        diff_models = set(file_path.split("/")[-2] for file_path in non_matching_files)  # noqa
        models_str = "\n - " + "\n - ".join(sorted(diff_models))
        raise ValueError(f"Some diff and their modeling code did not match. Models in diff:{models_str}")

    if skipped_models:
        console.print(
            f"[bold green]Skipped {len(skipped_models)} models and their dependencies that are not in the diff: "
            f"{', '.join(sorted(skipped_models))}[/bold green]"
        )
