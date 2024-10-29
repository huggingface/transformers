import argparse
import importlib.util
import logging
import os
import sys


class ImportModuleException(Exception):
    pass


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s - %(asctime)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    """
    Parse command line arguments for the benchmarking CLI.
    """
    parser = argparse.ArgumentParser(description="CLI for benchmarking the huggingface/transformers.")

    parser.add_argument(
        "branch",
        type=str,
        help="The branch name on which the benchmarking is performed.",
    )

    parser.add_argument(
        "commit_id",
        type=str,
        help="The commit hash on which the benchmarking is performed.",
    )

    parser.add_argument(
        "commit_msg",
        type=str,
        help="The commit message associated with the commit, truncated to 70 characters.",
    )

    args = parser.parse_args()

    return args.branch, args.commit_id, args.commit_msg


def import_from_path(module_name, file_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise ImportModuleException(f"failed to load python module: {e}")


if __name__ == "__main__":
    benchmarks_folder_path = os.path.dirname(os.path.realpath(__file__))

    branch, commit_id, commit_msg = parse_arguments()

    for entry in os.scandir(benchmarks_folder_path):
        try:
            if not entry.name.endswith(".py"):
                continue
            if entry.path == __file__:
                continue
            logger.debug(f"loading: {entry.name}")
            module = import_from_path(entry.name.split(".")[0], entry.path)
            logger.info(f"runnning benchmarks in: {entry.name}")
            module.run_benchmark(logger, branch, commit_id, commit_msg)
        except ImportModuleException:
            logger.debug("failed to load python module")
        except Exception as e:
            logger.debug(f"error running benchmarks for {entry.name}: {e}")
