from pathlib import Path

# root_path = r"I:\\BTU\\annotation\\"
# paths = [str(x) for x in Path(root_path).glob("**/*.txt")]

# for path in paths:
#     with open(root_path, 'r+',encoding='utf-8') as file:
#         lines = file.readlines()
#         for line in lines:
#             print(line.split('\t'))

root_path = r"./result.txt"
example_length = []
examples = {}
with open(root_path, 'r+', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        example = line.replace('\n','').split('\t')
        if len(example) == 1:
            examples[example[0]] = 0
        else:
            examples[example[0]] = 1

print(examples)

with open('train.txt','w+',encoding='utf-8') as f:
    for k,v in examples.items():
        f.write(str(k)+'\t'+str(v)+'\n')