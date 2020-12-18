from pathlib import Path

root_path = r"/home/wuyan/usr/material/bert_corpus"
path = [str(x) for x in Path(root_path).glob("**/*.txt")]


def merge_text(filenames):
    with open('result.txt', 'w') as outfile:
        for names in filenames:
            with open(names) as infile:
                outfile.write(infile.read())
            outfile.write("\n")
