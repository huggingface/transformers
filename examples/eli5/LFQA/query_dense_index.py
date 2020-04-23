import faiss
import json
import time 
import torch

from pprint import pprint
from train_eli5_retriever import *


model_file = 'retriever_models/embed_eli5_qa_512'
model_epoch = 4
print('----- loading model')
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-8_H-512_A-8")
bert_model = AutoModel.from_pretrained("google/bert_uncased_L-8_H-512_A-8").to('cuda:0')
qa_embedder = RetrievalQAEmbedder(bert_model, 512).to('cuda:0')
args = pickle.load(open('{}_args.pk'.format(model_file), 'rb'))
param_dict = torch.load('{}_{}.pth'.format(model_file, model_epoch))
qa_embedder.load_state_dict(param_dict['model'])

print('----- loading indexed wikipedia')
passages_text = json.load(open('../Wikipedia/wiki40b_docs_slice_0.json'))
passages_reps = torch.load('../Wikipedia/wiki40b_vecs_slice_0.pth')
for i in range(1, 3):
    print(i)
    passages_text += json.load(open('../Wikipedia/wiki40b_docs_slice_{}.json'.format(i)))
    passages_reps = torch.cat([passages_reps, torch.load('../Wikipedia/wiki40b_vecs_slice_{}.pth'.format(i))], dim=0)

print('----- making faiss index')
index_flat = faiss.IndexFlatIP(128)
res = faiss.StandardGpuResources()
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
st_time = time(); gpu_index_flat.add(passages_reps.numpy()); time() - st_time

def embed_question(q):
    with torch.no_grad():
        q_ids, q_mask, _, _ = make_batch([{'question': q, 'answer': 'A'}], tokenizer, max_len=128, max_start_idx=10000, device='cuda:0')
        q_reps = qa_embedder.embed_questions(q_ids, q_mask).cpu().type(torch.float).numpy()
    return q_reps

def retrieve_docs(q):
    scores, ids = gpu_index_flat.search(embed_question(q), 10)
    res = [passages_text[i] for i in list(ids[0])]
    return res

print('----- ready to go')

test_questions = [
    "Why can’t you use regular fuel in a car that “requires” premium?",
    "Why doesn't knocking on a door leave my hand a bruised and destroyed mess?",
    "If Tesla's inventions were as good for harnessing power as they are reported to be why are they not being used today?",
    'Why is the term "Patient Zero" instead of "Patient One?"',
]
for q in test_questions:
    print("\n\n---- Q ---- {}".format(q))
    for i, res_dct in enumerate(retrieve_docs(q)):
        print("---- Result {:2d}: {}".format(i, res_dct['_id']))
        print("-- {}".format(res_dct['content']))



