from pathlib import Path


# root_path = r"/home/wuyan/usr/material/bert_corpus"
# path = [str(x) for x in Path(root_path).glob("**/*.txt")]


def merge_text(filenames):
    with open('result.txt', 'w',encoding='utf-8') as outfile:
        for names in filenames:
            with open(names, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            outfile.write("\n")


if __name__ == '__main__':
    root_path = r"I:\\BTU\\annotation\\"
    path = [str(x) for x in Path(root_path).glob("**/*.txt")]
    merge_text(path)
