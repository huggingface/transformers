import re
import argparse

def parse_pytest_output(file_path):
    skipped_tests = {}
    skipped_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'^SKIPPED \[(\d+)\] (tests/[^/]+/[^:]+):(\d+): (.*)$', line)
            if match:
                skipped_count += 1
                print(match.groups())
                _, test_file, test_line, reason = match.groups()
                skipped_tests[reason] = skipped_tests.get(reason, []) + [(test_file, test_line)]
    return skipped_tests, skipped_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="file to parse")
    args = parser.parse_args()

    skipped_tests, skipped_count = parse_pytest_output(args.file)
    print("Number of skipped tests:", skipped_count)
    for k,v in sorted(skipped_tests.items(), key=lambda x:len(x[1])):
        print(f"{len(v):4} reason: {k}")


if __name__ == "__main__":
    main()