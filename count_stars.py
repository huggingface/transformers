import os
import time

if __name__ == "__main__":

    for i in range(86400):
        time.sleep(4)

        s1 = None
        s2 = None

        cmd = 'curl -s "https://api.github.com/repos/huggingface/transformers" | grep stargazers_count | cut -d : -f 2 | tee stars.txt'
        os.system(cmd)
        with open("stars.txt") as fp:
            s = fp.read()
            print(s)
            s1 = s[1:-1]
            if s2 != s1:
                print(f"🤗 stars ⭐: {s1}")
                s2 = s1
