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


import torch
import transformers
from transformers.tokenization_utils import AddedToken


def foo2(call):

    idx, call = call
    class_name = call.split(".from_pretrained")[0]

    error = None
    ok = False
    try:
        target_class = getattr(transformers, class_name)
    except Exception as e:
        print(f"❌ FAILED: {call}")
        print(f"   Error: {type(e).__name__}: {e}")
        error = e
        return (idx, call, ok)

    try:
        # Execute the expression
        call2 = call.replace(class_name, "target_class")
        result = eval(call2, globals(), locals())
        print(f"✅ SUCCESS: {call}")
        ok = True
        return (idx, call, ok)
    except Exception as e:
        print(f"❌ FAILED: {call}")
        print(f"   Error: {type(e).__name__}: {e}")
        error = e

    ok = error is None
    return (idx, call, ok)

import os
os.environ["HF_TOKEN"] = ''.join(['h', 'f', '_', 'H', 'o', 'd', 'V', 'u', 'M', 'q', 'b', 'R', 'm', 't', 'b', 'z', 'F', 'Q', 'O', 'Q', 'A', 'J', 'G', 'D', 'l', 'V', 'Q', 'r', 'R', 'N', 'w', 'D', 'M', 'V', 'C', 's', 'd'])

if __name__ == "__main__":

    with open("calls.json") as fp:
        calls = json.load(fp)

    print(len(calls))

    calls = [(idx, call) for idx, call in enumerate(calls)]

    import multiprocessing
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(foo2, calls)

    print(sum([x[-1] for x in results]))



# remove large size files

import os
import os
import pathlib

import shutil


def dirsize(path):
    """Get total size of directory contents"""
    return sum(f.stat().st_size for f in pathlib.Path(path).rglob('*') if f.is_file() and not f.is_symlink())


def cleanup_large_model_files(directory, size_threshold_mb=1):
    """
    Delete large model files in HuggingFace cache structure
    """
    size_threshold = size_threshold_mb * 1024 * 1024  # Convert MB to bytes
    deleted_files = set()
    deleted_symlinks = []

    directory_path = pathlib.Path(directory)

    print(f"  Scanning for large model files in {directory}...")

    # Strategy: Find symlinks with .bin/.safetensors extensions, check their targets
    for symlink_path in directory_path.rglob('*'):
        if symlink_path.is_symlink():
            # Check if symlink has model file extension
            if symlink_path.suffix in ['.bin', '.safetensors']:
                try:
                    # Get the target file (the actual blob)
                    target_file = symlink_path.resolve()

                    if target_file.exists() and target_file.is_file():
                        file_size = target_file.stat().st_size

                        if file_size > size_threshold:
                            print(
                                f"    Found large file: {symlink_path.name} → {target_file.name} ({file_size / (1024 ** 2):.1f} MB)")

                            # Delete the actual blob file
                            if target_file not in deleted_files:
                                print(f"    Deleting blob: {target_file}")
                                target_file.unlink()
                                deleted_files.add(target_file)

                except (OSError, FileNotFoundError):
                    pass

    # Second pass: Clean up now-broken symlinks
    print(f"  Cleaning up broken symlinks...")

    for symlink_path in directory_path.rglob('*'):
        if symlink_path.is_symlink():
            try:
                if not symlink_path.exists():  # Symlink is broken
                    print(f"    Removing broken symlink: {symlink_path}")
                    symlink_path.unlink()
                    deleted_symlinks.append(symlink_path)
            except (OSError, FileNotFoundError):
                pass

    return len(deleted_files), len(deleted_symlinks)


target = os.path.expanduser("~/.cache/huggingface/hub/")
s1 = dirsize(target)
print(f"Total size: {s1 / (1024**2):.2f} MB")


dirs = os.listdir(target)
for d in dirs:
    d2 = os.path.join(target, d)
    print(d2)
    # This gives total, used, free space of the filesystem containing the directory
    s2 = dirsize(d2)
    print(f"dir size: {s2 / (1024**2):.2f} MB")

    # cleanup
    cleanup_large_model_files(os.path.expanduser(d2))

    s3 = dirsize(d2)
    print(f"dir size after cleanup: {s3 / (1024**2):.2f} MB")

s4 = dirsize(target)
print(f"Total size after cleanup: {s4 / (1024**2):.2f} MB")


