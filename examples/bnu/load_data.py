from pathlib import Path

root_path = r"I:\\BTU\\annotation\\"
paths = [str(x) for x in Path(root_path).glob("**/*.txt")]


root_path = r"I:\\BTU\\annotation\\1.txt"

# with open(root_path, 'r+',encoding='utf-8') as file:
#     lines = file.readlines()
#     for line in lines:
#         print(line)
for path in paths:
    with open(root_path, 'r+',encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            print(line.split('\t'))
