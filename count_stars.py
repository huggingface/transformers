import os
import time

if __name__ == "__main__":

    for i in range(86400):
        time.sleep(1)

        cmd = 'curl -s "https://api.github.com/repos/huggingface/transformers" | grep stargazers_count | cut -d : -f 2 | tee stars.txt'
        os.system(cmd)
        with open("stars.txt") as fp:
            s = fp.read()
            s = s[1:-1]
            print(f"🤗 stars ⭐: {s}")
