import json
import sys
import time

import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk

wiki40b_file = sys.argv[1]
passage_length = int(sys.argv[2])

# python create_dense_index.py wiki40b.jsonl 100

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


es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])

index_config = {
  "settings": {
    "number_of_shards": 1,
    "analysis": {
      "analyzer": {
        "stop_standard": {
          "type": "standard",
          " stopwords": "_english_"
        }
      }
    }
  },
  "mappings": {
    "properties": {
        "title": {
          "type": "text",
          "analyzer": "standard",
          "similarity": "BM25",
        },
        "section": {
          "type": "text",
          "analyzer": "standard",
          "similarity": "BM25",
        },
        "content": {
          "type": "text",
          "analyzer": "stop_standard",
          "similarity": "BM25",
        },
      "popularity": {"type": "double"}
    }
  }
}

es_client.indices.create(index = 'english_wiki40b_passages_100w', body = index_config)
number_of_docs = 15500000

progress = tqdm.tqdm(unit="docs", total=number_of_docs)
successes = 0
for ok, action in streaming_bulk(
        client=es_client, index="english_wiki40b_passages_100w", actions=generate_paragraphs(wiki40b_file),
):
    progress.update(1)
    successes += ok
    
print("Indexed %d/%d documents" % (successes, number_of_docs))
