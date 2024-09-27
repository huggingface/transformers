import re
import argparse
import sys

def parse_pytest_output(file_path):
    """
    Parse the Pytest output and print the reasons for skipped tests.

    Args:
        file_path (str): The path to the Pytest output file.

    Returns:
        None
    """
    skipped_tests = {}
    skipped_count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(r'^SKIPPED \[(\d+)\] (tests/.*): (.*)$', line)
                if match:
                    skipped_count += 1
                    test_file, test_line, reason = match.groups()
                    skipped_tests[reason] = skipped_tests.get(reason, []) + [(test_file, test_line)]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    for k, v in sorted(skipped_tests.items(), key=lambda x: len(x[1])):
        print(f"{len(v):4} skipped because: {k}")
    print("Number of skipped tests:", skipped_count)

def parse_pytest_failure_output(file_path):
    """
    Parse the Pytest output and print the reasons for failed tests.

    Args:
        file_path (str): The path to the Pytest output file.

    Returns:
        None
    """
    failed_tests = {}
    failed_count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(r'^FAILED (tests/.*) - (.*): (.*)$', line)
                if match:
                    failed_count += 1
                    _, error, reason = match.groups()
                    failed_tests[reason] = failed_tests.get(reason, []) + [error]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    for k, v in sorted(failed_tests.items(), key=lambda x: len(x[1])):
        print(f"{len(v):4} failed because `{v[0]}` -> {k}")
    print("Number of failed tests:", failed_count)
    if failed_count > 0:
        sys.exit(1)

def parse_pytest_errors_output(file_path):
    """
    Parse the Pytest output and print the reasons for errored tests.

    Args:
        file_path (str): The path to the Pytest output file.

    Returns:
        None
    """
    error_tests = {}
    error_count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(r'^ERROR (tests/.*) - (.*): (.*)$', line)
                if match:
                    error_count += 1
                    _, test_error, reason = match.groups()
                    error_tests[reason] = error_tests.get(reason, []) + [test_error]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    for k, v in sorted(error_tests.items(), key=lambda x: len(x[1])):
        print(f"{len(v):4} errored out because of `{v[0]}` -> {k}")
    print("Number of errors:", error_count)
    if error_count > 0:
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="file to parse")
    parser.add_argument("--skip", action="store_true", help="show skipped reasons")
    parser.add_argument("--fail", action="store_true", help="show failed tests")
    parser.add_argument("--errors", action="store_true", help="show failed tests")
    args = parser.parse_args()

    if args.skip:
        parse_pytest_output(args.file)

    if args.fail:
        parse_pytest_failure_output(args.file)

    if args.errors:
        parse_pytest_errors_output(args.file)

if __name__ == "__main__":
    main()
