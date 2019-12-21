# bert+crf NER

## tags
Different from the run_ner.py already in the examples, we regard the tokens that are not at the starting position of a word to have the tag "X", which is in the labels set. E.g., fot the CoNLL03 task, the labels are ["X", "O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]. 