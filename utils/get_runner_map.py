import requests

response = requests.get("https://huggingface.co/datasets/hf-internal-testing/transformers_daily_ci/resolve/main/runner_map.json")
runner_map = response.json()

print(runner_map)
