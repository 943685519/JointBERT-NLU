from torch.nn import functional as F
from transformers import BertModel
from torch import nn
from torchcrf import CRF
import transformers
nltconfig = transformers.PretrainedConfig(name_or_path='bert-base-chinese')

class nluModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.pretrain_model_path)
        self.drop = nn.Dropout(p=0.2)
        self.num_intent_label = len(self.config.intent_vocab)
        self.num_slot_label = len(self.config.slot_vocab)
        self.hid_size = self.bert.config.hidden_size
        self.fc_intent = nn.Linear(self.hid_size,self.num_intent_label)
        self.fc_slot = nn.Linear(self.hid_size,self.num_slot_label)
        self.crf = CRF(self.config.num_slot,batch_first=True).cuda()


    def forward(self,input_ids,attention_mask,token_type_ids=None):
        out = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        seq_encoding,pooled_output = out[0],out[1]

        seq_encoding = self.drop(seq_encoding)       # bs,seq_len,hid_size
        pooled_output = self.drop(pooled_output)     # bs,hid_size

        intent_logits = self.fc_intent(pooled_output) #bs,num_intent_label
        slot_logits = self.fc_slot(seq_encoding)      # bs,seq_len,num_slot_label

        return intent_logits,slot_logits