import os
import requests

tests = os.getcwd()
model_tests = os.listdir(os.path.join(tests, "models"))
d1 = sorted(filter(os.path.isdir, os.listdir(tests)))
d2 = sorted(filter(os.path.isdir, [f"models/{x}" for x in model_tests]))
d1.remove("models")
d = d2 + d1

response = requests.get("https://huggingface.co/datasets/hf-internal-testing/transformers_daily_ci/resolve/main/runner_map.json")
runner_map = response.json()

for key in d:
    if key not in runner_map:
        runner_map[key] = {
            "single-gpu": "aws-g4dn-4xlarge-cache",
            "multi-gpu": "aws-g4dn-12xlarge-cache",
        }

print(runner_map)
