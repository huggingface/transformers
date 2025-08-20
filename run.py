import os
import sys

start = int(os.environ.get("START"))
end = int(os.environ.get("END"))

# collect the test
os.system('RUN_SLOW=1 python3 -m pytest tests/models/ -k "(not IntegrationTest) and (test_flash_attn_2 or _fa2_)" --collect-only --quiet | grep "::test" > test_names.txt')

# load the tests
with open("test_names.txt") as fp:
    test_names = fp.read().strip().split("\n")

print(len(test_names))
sys.stdout.flush()

# results saved to here
final_report_name = "summary_short.txt"
os.system(f"rm -rf {final_report_name}")
os.system(f'echo "" > {final_report_name}')
sys.stdout.flush()

# run the tests
for idx, test in enumerate(test_names[start:end]):

    print(f"test {idx}: {test}")
    sys.stdout.flush()
    sys.stdout.flush()

    cmd = f"HF_HOME=/mnt/cache RUN_SLOW=1 python3 -m pytest -v {test} --make-reports=tests_fa2"
    os.system(cmd)
    report_name = "reports/tests_fa2/summary_short.txt"
    if os.path.exists(report_name):
        with open(report_name, "r") as fp:
            report = fp.read().strip()
        for line in report.split("\n"):
            if line.startswith(("FAILED", "PASSED", "ERROR", "SKIPPED")):

                with open(final_report_name, "r") as fp:
                    saved_report = fp.read()
                    saved_tests = saved_report.strip().split("\n")

                if line.startswith(("FAILED", "ERROR")):
                    saved_tests = [line] + saved_tests
                elif line.startswith(("PASSED", "SKIPPED")):
                    saved_tests = saved_tests + [line]

                saved_report = "\n".join(saved_tests)
                with open(final_report_name, "w") as fp:
                    fp.write(saved_report)

                break
