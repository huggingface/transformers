import json
import sys
import time

from train_eli5_retriever import *

wiki40b_file = sys.argv[1]
model_file = sys.argv[2]
model_epoch = int(sys.argv[3])
passage_length = int(sys.argv[4])
batch_size = int(sys.argv[5])

# CUDA_VISIBLE_DEVICES=1 python create_dense_index.py ../Wikipedia/wiki40b.jsonl retriever_models/embed_eli5_qa_512 4 100 1024

def generate_paragraphs(articles_file, passage_len=100):
    f = open(articles_file)
    for i, line in enumerate(f):
        article = json.loads(line.strip())
        passage = []
        for j, section in enumerate(article['sections']):
            ct = 0
            for k, paragraph in enumerate(section['paragraphs']):
                for l, part in enumerate(paragraph):
                    if passage_len < 0:
                        part_id = json.dumps(
                            (article['article_title'][:128], section['title'][:128], ct)
                        )
                        doc = {
                            "_id": part_id,
                            "title": article['article_title'],
                            "section": section['title'],
                            "content": part,
                        }
                        ct += 1
                        yield doc
                    else:
                        passage += part.split()
                        while len(passage) > passage_len:
                            part_id = json.dumps(
                                (article['article_title'][:128], section['title'][:128], ct)
                            )
                            doc = {
                                "_id": part_id,
                                "title": article['article_title'],
                                "section": section['title'],
                                "content": ' '.join(passage[:passage_len]),
                            }
                            passage = passage[passage_len:]
                            ct += 1
                            yield doc
        if len(passage) > 8:
            part_id = json.dumps(
                (article['article_title'][:128], section['title'][:128], ct)
            )
            doc = {
                "_id": part_id,
                "title": article['article_title'],
                "section": section['title'],
                "content": ' '.join(passage),
            }
            yield doc
    f.close()


print('----- loading model')
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-8_H-512_A-8")
bert_model = AutoModel.from_pretrained("google/bert_uncased_L-8_H-512_A-8").to('cuda:0')
qa_embedder = RetrievalQAEmbedder(bert_model, 512).to('cuda:0')
args = pickle.load(open('{}_args.pk'.format(model_file), 'rb'))
param_dict = torch.load('{}_{}.pth'.format(model_file, model_epoch))
qa_embedder.load_state_dict(param_dict['model'])


n_batches = 0
n_saved = 0
docs = []
reps = []
print('----- encoding documents')
passages = []
st_time = time()
for par in generate_paragraphs(wiki40b_file, passage_length):
    passages += [par]
    if len(passages) == batch_size:
        pre_batch = [{'question': 'Q', 'answer': passage['content']} for passage in passages]
        _, _, a_ids, a_mask = make_batch(pre_batch, tokenizer, max_len=128, max_start_idx=100000, device='cuda:0')
        with torch.no_grad():
            a_reps = qa_embedder.embed_answerss(a_ids, a_mask).cpu().type(torch.float)
        docs += passages[:]
        reps += [a_reps]
        passages = []
        n_batches += 1
        if n_batches % 10 == 0:
            print("{:5d} batches: {:.2f}".format(n_batches, time() - st_time))
    if len(docs) > 3000000:
        print('writing slice', n_saved)
        json.dump(docs, open('../Wikipedia/wiki40b_docs_slice_{}.json'.format(n_saved), 'w'))
        all_reps = torch.cat(reps, dim=0)
        torch.save(all_reps, '../Wikipedia/wiki40b_vecs_slice_{}.pth'.format(n_saved))
        docs = []
        reps = []
        n_saved += 1


print('writing slice', n_saved)
json.dump(docs, open('../Wikipedia/wiki40b_docs_slice_{}.json'.format(n_saved), 'w'))
all_reps = torch.cat(reps, dim=0)
torch.save(all_reps, '../Wikipedia/wiki40b_vecs_slice_{}.pth'.format(n_saved))
docs = []
reps = []
