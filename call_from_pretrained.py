import json

# with open("repo_ids.json") as fp:
#     data = json.load(fp)
#
# calls = [x["call"] for x in data if x["call"] is not None]
#
# print(len(calls))
#
#
# calls = sorted(set(calls))
#
# print(len(calls))
#
#
# new_calls = []
#
# for call in calls:
#
#     if call is None:
#         continue
#
#     # if "tiny-random-BertForMaskedLM" in call:
#     #     breakpoint()
#
#
#     if ".from_pretrained" not in call:
#         continue
#
#     call = call.strip()
#
#     if call.startswith("return "):
#         call = call[len("return "):]
#
#     call = call.split(" = ")[-1]
#
#     class_name = call.split(".from_pretrained")[0]
#
#     if not call.startswith(f"{class_name}.from_pretrained"):
#         continue
#
#     new_calls.append(call)
#
# with open("calls.json", "w", encoding="utf-8") as fp:
#     json.dump(new_calls, fp, indent=4, ensure_ascii=False)
#
# exit(0)

import tempfile
import subprocess

def foo(call):

    idx, call = call

    class_name = call.split(".from_pretrained")[0]
    expr = f'import torch; from transformers.tokenization_utils import AddedToken; from transformers import {class_name}; torch_device = "cpu"; obj = {call}; print(obj.__class__); print("OK")'
    print(expr)

    result = subprocess.run(
        ["python", "-c", expr],
        # capture_output=True,
        text=True
    )
    ok = result.returncode == 0

    return (idx, call, ok)

if __name__ == "__main__":

    with open("calls.json") as fp:
        calls = json.load(fp)

    print(len(calls))

    # for x in calls[:10]:
    #     print(x)

    calls = [(idx, call) for idx, call in enumerate(calls)]

    import multiprocessing
    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(foo, calls[:8])

    breakpoint()

    # TODO: how to get error message and size?
    print(sum([x[-1] for x in results]))


    # for call in calls[:16]:
    #     foo(call)