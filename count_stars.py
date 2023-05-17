import os
import time
import requests


def foo(token):

    headers = None
    if token is not None:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    url = f"https://api.github.com/repos/huggingface/transformers"

    result = requests.get(url, headers=headers).json()
    count = result["stargazers_count"]
    print(f"⭐: {count}")


if __name__ == "__main__":
    for i in range(10000):
        time.sleep(5)
        foo(token=os.environ["ACCESS_REPO_INFO_TOKEN"])
