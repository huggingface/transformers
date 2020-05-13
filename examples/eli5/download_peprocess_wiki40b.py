import json
import tensorflow_datasets as tfds

from random import choice


def make_dict(art):
    title = art.strip().split('\n')[1]
    article = {'article_title': title, 'sections': []}
    section = {'title': '', 'paragraphs': []}
    lines = art.strip().split('\n')[2:]
    for i, line in enumerate(lines):
        if line == '_START_SECTION_' and i+1 < len(lines):
            if len(section['paragraphs']) > 0:
                article['sections'] += [section]
            section = {'title': lines[i+1], 'paragraphs': []}
        elif line == '_START_PARAGRAPH_' and i+1 < len(lines):
            section['paragraphs'] += [[par for par in lines[i+1].split('_NEWLINE_') if len(par.strip()) > 0]]
        else:
            continue
    if len(section['paragraphs']) > 0:
        article['sections'] += [section]
    return article


dataset = tfds.load('wiki40b/Wiki40B.en')
articles = []
for exple in dataset['train']:
    articles += [exple['text'].numpy().decode('utf-8')]

article_dicts = [make_dict(article) for article in articles]

f = open('wiki40b.jsonl', 'w')
for article in article_dicts:
    f.write(json.dumps(article) + '\n')

f.close()
