import os


target_files = ["failures_line.txt", "failures_long.txt", "failures_short.txt", "stats.txt", "summary_short.txt"]

report_directories = os.listdir("renamed_reports")

for test_job in report_directories:
    assert test_job.startswith("tests_") or test_job == "examples"
    report_files = os.listdir(os.path.join("renamed_reports", test_job))

    for target in target_files:
        prefix = target[:-len(".txt")] + "_"
        reports = []
        for fn in report_files:
            if fn.startswith(prefix):
                reports.append(fn)
        text = ""
        for fn in reports:
            p = os.path.join("renamed_reports", test_job, fn)
            with open(p) as fp:
                text += fp.read() + "\n\n"
        os.makedirs(f"combined_reports/{test_job}", exist_ok=True)
        with open(f"combined_reports/{test_job}/{prefix[:-1]}.txt", "w", encoding="UTF-8") as fp:
            fp.write(text)
