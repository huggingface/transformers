import http.client
import os

conn = http.client.HTTPSConnection("circleci.com")

pr_number = "43665"
payload = {"branch": f"pull/{pr_number}/head" , "parameters" :{"nightly": "false"}}
payload = str(payload).replace("'", '"').replace('"false"', 'false')
print(payload)

print(len(os.environ.get("Circle-Token", "")))
print(os.environ.get("Circle-Token", "")[:5])

headers = {
    'content-type': "application/json",
    'Circle-Token': os.environ.get("CIRCLECI_TOKEN", ""),
}

conn.request("POST", "/api/v2/project/gh/huggingface/transformers/pipeline", payload, headers)

res = conn.getresponse()
data = res.read()