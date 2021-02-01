import torch
from torch.utils.data import Dataset,DataLoader
from config import config

class nluDataset(Dataset):
    def __init__(self,config,file):
        self.config = config
        self.data = self.make_dataset(file)

    def make_dataset(self,file):
        dataset = []
        with open(file,'r',encoding='utf8') as f:
            lines = [line.split('\t') for line in f.readlines()]
            for text,intent,slot in lines:
                text = self.config.tokz.convert_tokens_to_ids(list(text)[:self.config.max_len])
                intent = self.config.intent_vocab[intent]
                slot = [self.config.slot_vocab[i] for i in slot.split()]

                dataset.append([
                    [self.config.tokz.cls_token_id] + text + [self.config.tokz.sep_token_id],
                    intent,
                    [self.config.tokz.pad_token_id] + slot + [self.config.tokz.pad_token_id],
                    self.config.tokz.create_token_type_ids_from_sequences(token_ids_0=text)
                ])
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text,intent,slot,token_type = self.data[idx]
        return {'text':text, 'intent':intent, 'slot':slot, 'token_type':token_type}

class PinnedBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, k):
        return self.data[k]

    def pin_memory(self):
        for k in self.data.keys():
            self.data[k] = self.data[k].pin_memory()
        return self

class PadBatchData:
    def __init__(self,config):
        self.pad_id = config.pad_idx

    def __call__(self,batch):
        res = dict()
        max_len = max([len(i['text']) for i in batch])
        res['text'] = torch.LongTensor( [i['text'] + [self.pad_id]*(max_len-len(i['text'])) for i in batch] )
        res['intent'] = torch.LongTensor( [i['intent'] for i in batch])
        res['slot'] = torch.LongTensor([i['slot'] + [self.pad_id]*(max_len-len(i['slot'])) for i in batch])
        res['mask'] = torch.LongTensor([[1]*len(i['text'])+[self.pad_id]*(max_len-len(i['text'])) for i in batch])
        res['token_type'] = torch.LongTensor([i['token_type'] + [self.pad_id]*(max_len-len(i['token_type'])) for i in batch])
        # return PinnedBatch(res)
        return PinnedBatch(res)

if __name__ == '__main__':
    config = config()
    dataset = nluDataset(config,'data/test.txt')
    for i in dataset:
        print(i)
        break
    # print(config.tokz.decode([101, 7944, 2126, 2769, 6206, 4381, 704, 1744, 6496, 3470, 102]))
#     for i in train_dataset:
#         text = [config.tokz.decode(w) for w in i['text']]
#         intent = [config.idx2intent[w] for w in i['intent']]
#         slot = [config.idx2slot[w] for w in i['slot']]
#         print(text,intent,slot)
#