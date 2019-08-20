import json
from pyltp import SentenceSplitter

def get_corpus(data_path = '/data/share/zhanghaipeng/data/chuangtouribao/raw_data.json', save_path = './corpus.txt'):
    writer = open(save_path, 'w')
    with open(data_path, 'r') as reader:
        for i,line in enumerate(reader):
            example = json.loads(json.loads(line)['input'])
            for data in example:
                sents = list(SentenceSplitter.split(data['data']))
                sents_filter = [sent for sent in sents if len(sent) != 0]
                writer.write('\n'.join(sents_filter))
            writer.write('\n\n')
    writer.close()
if __name__ == '__main__':
    get_corpus()
