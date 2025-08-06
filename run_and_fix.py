import os
import json

fn = "new_failures_with_bad_commit_grouped_by_authors.json"

with open(fn) as fp:
    data = json.load(fp)

tests = []
for model in data["ydshieh"]:
    for test in data["ydshieh"][model]["single-gpu"]:
        tests.append(test["test"])

print(len(tests))

for test in tests[:3]:
    print(test)
    command = f"HF_HOME=/mnt/cache RUN_SLOW=1 python3 -m pytest -v {test}"
    os.system(command)
