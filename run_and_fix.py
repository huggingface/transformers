import os
import json

fn = "new_failures_with_bad_commit_grouped_by_authors.json"

with open(fn) as fp:
    data = json.load(fp)

tests = []
for model in data["ydshieh"]:
    for test in data["ydshieh"][model]["single-gpu"]:
        tests.append(test["test"])

fn2 = "fixed.json"

with open(fn2) as fp:
    fixed = json.load(fp)

to_fix = [x for x in tests if x not in fixed]

print(len(tests))
print(len(fixed))
print(len(to_fix))

fn3 = "to_fix.json"

with open(fn3, "w") as fp:
    json.dump(to_fix, fp, indent=4)

fn4 = "to_fix.txt"

with open(fn4, "w") as fp:
    fp.write("\n".join(to_fix))

for test in to_fix:
    print(test)
    command = f"HF_HOME=/mnt/cache RUN_SLOW=1 python3 -m pytest -v {test}"
    os.system(command)
