import json
from collections import Counter
from tqdm import tqdm

def get_target_fields(annotation):
    return_info = 0
    return_anno = {}
    target_labels = ['融资主体','投资机构','投融资金额','融资轮次']
    old_label = annotation['label'][0]
    if old_label.find(target_labels[1]) != -1:
        new_label = target_labels[1]
    else:
        new_label = old_label[old_label.find('-')+1:]
    
    return_anno['label'] = new_label
    return_anno['position'] = annotation['position']
    
    if new_label in target_labels:
        return_info = 1
    return return_info, return_anno

def get_data(data_dir = '/data/share/zhanghaipeng/data/chuangtouribao/event/',data_path='raw_data.json',save_path = 'data.json'):
    writer = open(data_dir+save_path, 'a')
    with open(data_dir+data_path, 'r') as reader:
        textlines = reader.readlines()
        total_num = len(textlines)
        for i,line in enumerate(textlines):
            print(i,total_num)
            article = json.loads(json.loads(line)['input'])
            for para in article:
                if 'annotation' in para.keys():
                    annotations = para['annotation']
                    data = para['data']
                    group_id = Counter([anno['group_id'] for anno in annotations])
                    for key in group_id.keys():
                        event_dict = {}
                        event_dict['text'] = data
                        event_dict['anno_list'] = []
                        event = [anno for anno in annotations if anno['group_id'] == key]
                        for anno in event:
                            info, new_anno = get_target_fields(anno)
                            if info:
                                event_dict['anno_list'].append([new_anno['label'], new_anno['position']])
                        writer.write(json.dumps(event_dict,ensure_ascii=False)+'\n')
    writer.close()
                        

if __name__ == '__main__':
    get_data()
