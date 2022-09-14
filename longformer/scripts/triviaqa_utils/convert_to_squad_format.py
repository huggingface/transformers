from . import file_utils
from . import dataset_utils
import os
from tqdm import tqdm
import random
import nltk
import argparse


def get_text(qad, domain):
    local_file = os.path.join(args.web_dir, qad['Filename']) if domain == 'SearchResults' else os.path.join(args.wikipedia_dir, qad['Filename'])
    return file_utils.get_file_contents(local_file, encoding='utf-8')


def select_relevant_portion(text):
    paras = text.split('\n')
    selected = []
    done = False
    for para in paras:
        # nltk is slow, but we have to use its word tokenizer for the distant supervision matching to work
        # TODO: try both see which one works better
        # words = para.split()
        # extra_words = args.max_num_tokens - len(selected)
        # selected.extend(words[:extra_words])
        # if len(selected) >= args.max_num_tokens:
        #     break
        sents = sent_tokenize.tokenize(para)
        for sent in sents:
            words = nltk.word_tokenize(sent)
            for word in words:
                selected.append(word)
                if len(selected) >= args.max_num_tokens:
                    done = True
                    break
            if done:
                break
        if done:
            break
        selected.append('\n')
    st = ' '.join(selected).strip()
    return st


def add_triple_data(datum, page, domain):
    qad = {'Source': domain}
    for key in ['QuestionId', 'Question', 'Answer']:
        if key == 'Answer' and key not in datum:
            qad[key] = {'NormalizedAliases': []}
            qid = datum['QuestionId']
            print(f'qid: {qid} does not have an answer.')
        else:
            qad[key] = datum[key]
    for key in page:
        qad[key] = page[key]
    return qad


def get_qad_triples(data):
    qad_triples = []
    for datum in data['Data']:
        for key in ['EntityPages', 'SearchResults']:
            for page in datum.get(key, []):
                qad = add_triple_data(datum, page, key)
                qad_triples.append(qad)
    return qad_triples


def convert_to_squad_format(qa_json_file, squad_file):
    qa_json = dataset_utils.read_triviaqa_data(qa_json_file)
    qad_triples = get_qad_triples(qa_json)
    random.seed(args.seed)
    random.shuffle(qad_triples)

    data = []
    for qad in tqdm(qad_triples):
        qid = qad['QuestionId']

        text = get_text(qad, qad['Source'])
        selected_text = select_relevant_portion(text)

        question = qad['Question']
        para = {'context': selected_text, 'qas': [{'question': question, 'answers': []}]}
        data.append({'paragraphs': [para]})
        qa = para['qas'][0]
        qa['id'] = dataset_utils.get_question_doc_string(qid, qad['Filename'])
        qa['qid'] = qid

        answers_in_doc = dataset_utils.answer_index_in_document(qad['Answer'], selected_text)
        qa['answers'] = answers_in_doc
        # We want all answers in the document, not just the first answer
        # if index == -1:
        #     if qa_json['Split'] == 'train':
        #         continue
        # else:
        #     qa['answers'].append({'text': ans_string, 'answer_start': index})

        # This doesn't fit the squad format, but we need it for evaluation
        qa['aliases'] = qad['Answer']['NormalizedAliases']

        if qa_json['Split'] == 'train' and len(data) >= args.sample_size and qa_json['Domain'] == 'Web':
            break

        if len(data) >= args.sample_size:
            break

    squad = {'data': data, 'version': qa_json['Version']}
    file_utils.write_json_to_file(squad, squad_file)
    print('Added', len(data))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--triviaqa_file', help='Triviaqa file')
    parser.add_argument('--squad_file', help='Squad file')
    parser.add_argument('--wikipedia_dir', help='Wikipedia doc dir')
    parser.add_argument('--web_dir', help='Web doc dir')

    parser.add_argument('--seed', default=10, type=int, help='Random seed')
    parser.add_argument('--max_num_tokens', default=800, type=int, help='Maximum number of tokens from a document')
    parser.add_argument('--sample_size', default=8000000000000, type=int, help='Random seed')
    parser.add_argument('--tokenizer', default='tokenizers/punkt/english.pickle', help='Sentence tokenizer')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    sent_tokenize = nltk.data.load(args.tokenizer)
    convert_to_squad_format(args.triviaqa_file, args.squad_file)
