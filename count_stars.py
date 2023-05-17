import os
import time
import requests
from datetime import datetime

def foo(token, old):

    headers = None
    if token is not None:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    url = f"https://api.github.com/repos/huggingface/transformers"

    result = requests.get(url, headers=headers).json()
    count = result["stargazers_count"]
    # print(f"⭐: {count}")

    now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    if count != old:
        print("-" * 40)
    print(f"⭐: {count} | {now}")

    return count


if __name__ == "__main__":
    old = None
    for i in range(10000):
        time.sleep(10)
        foo(token=os.environ["ACCESS_REPO_INFO_TOKEN"], old=old)
