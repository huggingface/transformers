import re
doc_tokens = ["this", "is", "a", "test", ".", "I", "have", "a", "new", "question", "!", "please", "answer", ".", "...", "a", "word"]
map_dict = {}
for (i, n) in enumerate(doc_tokens):
    if n !=-1:
        map_dict[n] = i

print(idx)
exit()
idx = [i for i, n in enumerate(doc_tokens) if n == "." or n == "!" or n == "?"]
if idx[-1] != len(doc_tokens):
    idx = idx +[len(doc_tokens)]
sent_ends = list(map(lambda x: x + 1, idx))
sent_starts = [0] + sent_ends
sent_spans = list(zip(sent_starts, sent_ends))
for span in sent_spans:
    (start,end) = span
    print(" ".join(doc_tokens[start:end]))