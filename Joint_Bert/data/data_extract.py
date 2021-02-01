import json
import os
import random

random.seed(13)

data_file = 'data.json'

class data_extract:
    def __init__(self,file):
        self.file = file
        self.data = self.function()
        if('intent_label.txt' not in os.listdir('../data')):
            self.save_intent_label(self.data[1])
        if('slot_label.txt' not in os.listdir('../data')):
            self.save_slot_label(self.data[2])

    def function(self):
        with open(self.file,encoding='utf8') as f:
            data = json.load(f)
            intent_label = set()
            slot_label = set()
            slot_label.add('o')
            total_data = []

            for sent in data:
                sample = []
                text = sent['text']
                domain = sent['domain']
                intent = str(sent['intent'])
                slot = sent['slots']

                # 更新label集合
                if intent != 'nan':
                    intent_ = '{}@{}'.format(domain, intent)
                    intent_label.add(intent_)
                # 构建BIO标记
                if len(slot) != 0:
                    bio = ['o'] * len(text)
                    for k, v in slot.items():
                        idx = text.index(v[0])
                        sign = 'B-' + k
                        bio[idx] = sign
                        slot_label.add(sign)
                        for i in range(idx + 1, idx + len(v)):
                            sign = 'I-' + k
                            bio[i] = sign
                            slot_label.add(sign)
                if len(slot) == 0:
                    bio = ['o'] * len(text)
                sample = [text, intent_, bio]
                total_data.append(sample)
        return total_data,intent_label,slot_label

    def save_datafile(self,data_name,total_data):
        file = data_name + '.txt'
        with open(file, encoding='utf-8', mode='w') as f:
            for sample in total_data:
                text = sample[0] + '\t' + sample[1] + '\t' + ' '.join(sample[2])
                text = text.lower()
                f.write(text + '\n')
        print('write {}_data file.'.format(data_name))

    def save_slot_label(self,slot_label):
        with open('slot_label.txt', encoding='utf-8', mode='w') as f:
            for word in slot_label:
                word = word.lower()
                f.write(word + '\n')
        print('write <slot_label.txt> file.')

    def save_intent_label(self,intent_label):
        with open('intent_label.txt', encoding='utf-8', mode='w') as f:
            for word in intent_label:
                word = word.lower()
                f.write(word + '\n')
        print('write <intent_label.txt> file.')

def train_and_test_split(total_data):
    num = len(total_data)
    test_ids = random.sample(range(num),578)
    test_data = [data for i,data in enumerate(total_data) if i in test_ids]
    train_data = [data for i,data in enumerate(total_data) if i not in test_ids]
    return train_data,test_data

if __name__ == '__main__':
    data_extracter = data_extract(data_file)
    total_data = data_extracter.data[0]
    train_data,test_data = train_and_test_split(total_data)
    data_extracter.save_datafile('train',train_data)
    data_extracter.save_datafile('test',test_data)
