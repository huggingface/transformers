import os

pr_number = ""

pr = os.environ.get("CIRCLE_PULL_REQUEST", "")
if len(pr) > 0:
    pr_number = pr.split("/")[-1]
if pr_number == "":
    pr = os.environ.get("CIRCLE_BRANCH", "")
    if pr.startswith("pull/"):
        pr_number = "".join(pr.split("/")[1:2])

print(pr_number)
