from transformers import BertModel,BertTokenizer
import torch
import random
class config:
    def __init__(self,use_crf = True):
        torch.manual_seed(13)
        torch.cuda.manual_seed(13)
        random.seed(13)

        self.use_crf = use_crf

        self.pretrain_model_path = 'bert-base-chinese'
        self.train_file = 'data/train.txt'
        self.test_file = 'data/test.txt'

        self.lr = 8e-6
        self.bs = 30
        self.batch_split = 2
        self.eval_step = 40
        self.num_intent = 47
        self.num_slot = 122

        self.pad_idx = 0
        self.max_len = 90

        self.slot_vocab = self.build_vocab('data/slot_label.txt')
        self.intent_vocab = self.build_vocab('data/intent_label.txt')

        self.idx2slot = {v:k for k,v in self.slot_vocab.items()}
        self.idx2intent = {v: k for k, v in self.intent_vocab.items()}

        self.tokz = BertTokenizer.from_pretrained(self.pretrain_model_path)
        self.bert = BertModel.from_pretrained(self.pretrain_model_path)

    def build_vocab(self,file):
        with open(file,encoding='utf-8') as f:
            lines = f.readlines()
            vocab = {word.strip():idx for idx,word in enumerate(lines)}
        return vocab

