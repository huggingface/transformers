import argparse
import json
import spacy
import nltk
import torch
from torch.utils import data
from itertools import permutations 

# import seaborn as sns
# import matplotlib.pyplot as plt

_spacy_ner_types = {
    'PERSON':0,
    'NORP':1,
    'FAC':2,
    'ORG':3,
    'GPE':4,
    'LOC':5,
    'PRODUCT':6,
    'EVENT':7,
    'WORK_OF_ART':8,
    'LAW':9,
    'LANGUAGE':10,
    'DATE':11,
    'TIME':12,
    'PERCENT':13,
    'MONEY':14,
    'QUANTITY':15,
    'ORDINAL':16,
    'CARDINAL':17
}
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

'''
Reading, parsing SQuAD dataset.
'''
def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return list()
    tokens = text.split()
    return tokens

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def read_squad_data(data_file):
    with open(data_file, "r", encoding='utf-8') as open_file:
        read_file = json.load(open_file)["data"]

    squad_list = list()

    max_sentence_len = 0

    for entry in read_file:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"].lower()
            document = list()
            sentence_list = list()
            char_to_word_offset = list()
            prev_is_whitespace = True
            # construct char index -> token index list
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        document.append(c)
                    else:
                        document[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(document) - 1)

            # spacy has 18 named entity types
            sentence_list = split_into_sentences(paragraph_text)
            max_sentence_len = max(max_sentence_len, len(sentence_list))
            nltk_tokenized = [list(map(lambda x: nltk.word_tokenize(x), sentence_list))]

            for qa in paragraph["qas"]:
                if not bool(qa["is_impossible"]):
                    qas_id = qa["id"]
                    question_text =nltk.word_tokenize(qa["question"].lower())
                    answer = qa["answers"][0]
                    answer_text = answer["text"].lower()
                    answer_length = len(answer_text)

                    answer_offset = answer["answer_start"]                
                    answer_start = char_to_word_offset[answer_offset]
                    answer_end = char_to_word_offset[answer_offset + answer_length - 1]
                    actual_text = " ".join(document[answer_start:(answer_end + 1)]).lower()
                    cleaned_answer_text = " ".join(whitespace_tokenize(answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        continue

                    qa_pair = {
                        'id': qas_id,
                        'question': question_text,
                        'document': nltk_tokenized,
                        'sentence_list': sentence_list,
                        'answer_text': nltk.word_tokenize(answer_text.lower()),
                        'answer_start': answer_start,
                        'answer_end': answer_end
                    }
                    squad_list.append(qa_pair)

    return squad_list


'''
Create adjacency matrix
'''



class SquadDataset(data.Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        self.squad_list = read_squad_data(data_file)

    def __len__(self):
        return len(self.squad_list)

    def __getitem__(self, idx):
        return self.squad_list[idx]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=None, \
        type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    args = parser.parse_args()

    squad_dataset = SquadDataset(args.data_file)

    for i in range(len(squad_dataset)):
        with open("../squad_data/preprocessed_vn/train_text/" + str(squad_dataset[i]['id']) + '.json', 'w') as f:
            json.dump(squad_dataset[i], f)
        

if __name__ == "__main__":
    main()
