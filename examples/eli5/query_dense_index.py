import faiss
import json
import shelve
import time 
import torch

from pprint import pprint
from train_eli5_retriever import *
make_batch_retriever = make_batch

from train_eli5_finetuned_s2s import *

model_file = 'retriever_models/embed_eli5_qa_512'
model_epoch = 4
st_time = time()

print('----- loading indexed wikipedia', time() - st_time)
make_shelve = False
if make_shelve:
    passages_shlf = shelve.open('../Wikipedia/wiki40b_idtodocs', writeback=True)
    passages_ids = []
    passages_reps = torch.Tensor(0, 128).type(torch.float)
    for i in range(6):
        print('-- loading docs', i, time() - st_time)
        passages_docs =  json.load(open('../Wikipedia/wiki40b_docs_slice_{}.json'.format(i)))
        print('-- adding docs to shelve', i, time() - st_time)
        for dct in passages_docs:
            passages_shlf[dct['_id']] = dct['content']
        passages_ids += [dct['_id'] for dct in passages_docs]
        passages_shlf.sync()
        print('-- loading vecs', i, time() - st_time)
        passages_reps = torch.cat([passages_reps, torch.load('../Wikipedia/wiki40b_vecs_slice_{}.pth'.format(i))], dim=0)
    json.dump(passages_ids, open('../Wikipedia/wiki40b_idtodocs_keys.json', 'w'))
else:
    print('-- opening shelf', time() - st_time)
    passages_shlf = shelve.open('../Wikipedia/wiki40b_idtodocs', writeback=False)
    print('-- loading passage ids', time() - st_time)
    passages_ids = json.load(open('../Wikipedia/wiki40b_idtodocs_keys.json'))
    passages_reps = torch.Tensor(0, 128).type(torch.float)
    for i in range(6):
        print('-- loading vecs', i, time() - st_time)
        passages_reps = torch.cat([passages_reps, torch.load('../Wikipedia/wiki40b_vecs_slice_{}.pth'.format(i))], dim=0)

print('----- loading model', time() - st_time)
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-8_H-512_A-8")
bert_model = AutoModel.from_pretrained("google/bert_uncased_L-8_H-512_A-8").to('cuda:0')
qa_embedder = RetrievalQAEmbedder(bert_model, 512).to('cuda:0')
args = pickle.load(open('{}_args.pk'.format(model_file), 'rb'))
param_dict = torch.load('{}_{}.pth'.format(model_file, model_epoch))
qa_embedder.load_state_dict(param_dict['model'])

print('----- making faiss index', time() - st_time)
index_flat = faiss.IndexFlatIP(128)
res = faiss.StandardGpuResources()
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
st_time = time(); gpu_index_flat.add(passages_reps.numpy()); time() - st_time

def embed_question(q):
    with torch.no_grad():
        q_ids, q_mask, _, _ = make_batch_retriever([{'question': q, 'answer': 'A'}], tokenizer, max_len=128, max_start_idx=10000, device='cuda:0')
        q_reps = qa_embedder.embed_questions(q_ids, q_mask).cpu().type(torch.float).numpy()
    return q_reps

def retrieve_docs(q):
    scores, ids = gpu_index_flat.search(embed_question(q), 10)
    res = [passages_ids[i] for i in list(ids[0])]
    return res

print('----- ready to go', time() - st_time)

test_questions = [
    "Why can’t you use regular fuel in a car that “requires” premium?",
    "Why doesn't knocking on a door leave my hand a bruised and destroyed mess?",
    "If Tesla's inventions were as good for harnessing power as they are reported to be why are they not being used today?",
    'Why is the term "Patient Zero" instead of "Patient One?"',
    "If our immune system's response to infection includes raising our temperature, why do we fight that with fever-reducing medicines?",
    "How can different animals perceive different colors?",
    "Why are flutes classified as woodwinds when most of them are made out of metal?",
    "How can we set a date to the beginning or end of an artistic period? Doesn't the change happen gradually?",
    "Why do adults like drinking coffee when it tastes so bad?",
    "What happens when wine ages? How does it make the wine taste better?",
    "What exactly are vitamins?",
    "Why were the main predators in New Zealand large birds, compared to other places in the world?",
    "What's the difference between viruses and bacteria?",
    "If an animal is an herbivore (only eats plants) , where does it get the protein that it needs to survive if it only eats grass?",
    "How do you make chocolate?",
]

for q in test_questions:
    print("\n\n---- Q ---- {}".format(q))
    for i, p_id in enumerate(retrieve_docs(q)):
        print("---- Result {:2d}: {}".format(i, p_id))
        print("-- {}".format(passages_shlf[p_id]))

s2s_model, s2s_tokenizer, args = load_saved(
        '{}_args.pk'.format('models/bart_ft_wiki_qda'),
        '{}_{}.pth'.format('models/bart_ft_wiki_qda', 11),
)
s2s_model = s2s_model.to('cuda:0')

print('s2s model ready to go')

for q in test_questions:
    d = '<P> ' + ' <P> '.join([passages_shlf[p_id] for p_id in retrieve_docs(q)])
    pred = answer_example(
            {'id': 'NULL', 'question':q, 'document':d}, s2s_model, s2s_tokenizer, args,
            min_length=128,
            max_length=256,
            n_beams=4,
            n_samples=4, verbose=True
        )
    
while True:
    q = input()
    if q.strip() == 'QUIT':
        break
    else:
        d = '<P> ' + ' <P> '.join([passages_shlf[p_id] for p_id in retrieve_docs(q)])
    pred = answer_example(
            {'id': 'NULL', 'question':q, 'document':d}, s2s_model, s2s_tokenizer, args,
            min_length=128,
            max_length=256,
            n_beams=4,
            n_samples=4, verbose=True
        )
        


passages_shlf.close()

