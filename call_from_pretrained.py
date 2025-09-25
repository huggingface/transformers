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

# exit(0)

with open("calls.json") as fp:
    calls = json.load(fp)

print(len(calls))

for call in calls:

    class_name = call.split(".from_pretrained")[0]
    expr = f'from transformers import {class_name}; torch_device = "cpu"; obj = {call}; print(obj.__class__); print("OK")'
    print(expr)
    with open("file.py", "w", encoding="utf-8") as fp:
        fp.write(expr)

    import os
    os.system("python file.py")
