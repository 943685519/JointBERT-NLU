import torch
import torch.nn as nn
from dataset import nluDataset,PadBatchData,PinnedBatch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer:
    def __init__(self,config=None,model=None,train_dataset=None,valid_dataset=None,device=torch.device('cuda')):
        self.config = config
        self.device = device

        self.step=0

        self.model = model.to(device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.pad_idx,reduction='none').to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config.lr,weight_decay=0.01)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.bs, pin_memory=True,
                                           shuffle=True, collate_fn=PadBatchData(self.config))
        self.valid_dataloader = DataLoader(valid_dataset,batch_size=self.config.bs, pin_memory=True,
                                           shuffle=True,collate_fn=PadBatchData(self.config))

        self.train_writer = SummaryWriter('train')
        self.test_writer = SummaryWriter('test')

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self,state_dict):
        self.model.load_state_dict(state_dict)

    def slot_loss(self,slot_logits,slot_labels,mask):
        if self.config.use_crf:
            batch_slot_loss = -1 * self.model.crf(slot_logits, slot_labels, mask=mask.byte(), reduction='mean')
        else:
            batch_slot_loss = self.criterion(slot_logits.view(-1,slot_logits.size(-1)), slot_labels.view(-1)).mean()
        return batch_slot_loss

    def train(self, epoch):
        self.model.train()

        intent_loss, slot_loss, intent_acc, slot_acc, seq_slot_acc, step_count = 0, 0, 0, 0, 0, 0
        total = len(self.train_dataloader)

        TQDM = tqdm(enumerate(self.train_dataloader), desc='Train (epoch #{})'.format(epoch),
                    dynamic_ncols=True, total=total,mininterval = 1.5)

        for i, data in TQDM:
            text = data['text'].to(self.device, non_blocking=True)  #batch_size * seq_len
            intent_labels = data['intent'].to(self.device, non_blocking=True) #batch_size
            slot_labels = data['slot'].to(self.device, non_blocking=True) #batch_size * seq_len
            mask = data['mask'].to(self.device, non_blocking=True) #batch_size * seq_len
            token_type = data['token_type'].to(self.device, non_blocking=True) #batch_size * seq_len

            intent_logits, slot_logits = self.model(input_ids=text, attention_mask=mask, token_type_ids=token_type)

            batch_intent_loss = self.criterion(intent_logits, intent_labels).mean()
            batch_slot_loss = self.slot_loss(slot_logits,slot_labels,mask)

            slot_mask = 1 - slot_labels.eq(0).float()
            batch_loss = batch_intent_loss + batch_slot_loss

            batch_intent_acc = (torch.argmax(intent_logits, dim=-1) == intent_labels).float().mean()
            batch_slot_acc = (torch.argmax(slot_logits, dim=-1) == slot_labels)
            batch_slot_acc = torch.sum(batch_slot_acc * slot_mask) / torch.sum(slot_mask)

#求sequence_acc
            match = (torch.argmax(slot_logits,dim=-1) == slot_labels).float()
            slot_num = torch.sum(slot_mask,dim=-1)
            match_num = torch.sum(match*slot_mask,dim=-1)
            result = torch.sum((slot_num == match_num).float())
            batch_seq_slot_acc = result / slot_labels.size(0)

            full_loss = batch_loss / self.config.batch_split
            full_loss.backward()

            intent_loss += batch_intent_loss.item()
            slot_loss += batch_slot_loss.item()
            intent_acc += batch_intent_acc.item()
            slot_acc += batch_slot_acc.item()
            seq_slot_acc += batch_seq_slot_acc.item()
            step_count += 1

            self.step+=1
            if (i + 1) % self.config.batch_split == 0:
                # update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                intent_loss /= step_count
                slot_loss /= step_count
                intent_acc /= step_count
                slot_acc /= step_count
                seq_slot_acc /= step_count

                self.train_writer.add_scalar('loss/intent_loss',intent_loss,self.step)
                self.train_writer.add_scalar('loss/slot_loss',slot_loss,self.step)
                self.train_writer.add_scalar('acc/intent_acc', intent_acc, self.step)
                self.train_writer.add_scalar('acc/slot_acc', intent_acc, self.step)
                self.train_writer.add_scalar('acc/sequence_acc', seq_slot_acc,self.step)
                # self.train_writer.add_scalar('lr', lr, self.step)

                TQDM.set_postfix({'intent_loss': intent_loss, 'intent_acc': intent_acc, 'slot_loss': slot_loss,
                                  'slot_acc': slot_acc,'seq_acc':seq_slot_acc})
                intent_loss, slot_loss, intent_acc, slot_acc, seq_slot_acc, step_count = 0, 0, 0, 0, 0, 0

                if epoch%5==0:
                    self.test(epoch,epoch)


    def test(self, epoch, step):
        self.model.eval()
        with torch.no_grad():
            dev_intent_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            dev_slot_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            dev_intent_acc = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            dev_slot_acc = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            dev_seq_acc = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            count = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            for data in self.valid_dataloader:
                text = data['text'].to(self.device, non_blocking=True)
                intent_labels = data['intent'].to(self.device, non_blocking=True)
                slot_labels = data['slot'].to(self.device, non_blocking=True)
                mask = data['mask'].to(self.device, non_blocking=True)
                token_type = data['token_type'].to(self.device, non_blocking=True)

                intent_logits, slot_logits = self.model(input_ids=text, attention_mask=mask, token_type_ids=token_type)

                batch_intent_loss = self.criterion(intent_logits, intent_labels)
                batch_slot_loss = self.criterion(slot_logits.view(-1, slot_logits.shape[-1]), slot_labels.view(-1))
                slot_mask = 1 - slot_labels.eq(self.config.tokz.pad_token_id).float()
                batch_slot_loss = (batch_slot_loss * slot_mask.view(-1)).view(text.shape[0], -1).sum(
                    dim=-1) / slot_mask.sum(dim=-1)

                dev_intent_loss += batch_intent_loss.sum()
                dev_slot_loss += batch_slot_loss.sum()

                batch_intent_acc = (torch.argmax(intent_logits, dim=-1) == intent_labels).sum()
                batch_slot_acc = (torch.argmax(slot_logits, dim=-1) == slot_labels)
                batch_slot_acc = torch.sum(batch_slot_acc * slot_mask, dim=-1) / torch.sum(slot_mask, dim=-1)
# 求sequence_acc
                match = (torch.argmax(slot_logits, dim=-1) == slot_labels).float()
                slot_num = torch.sum(slot_mask, dim=-1)
                match_num = torch.sum(match * slot_mask, dim=-1)
                batch_seq_slot_acc = torch.sum((slot_num == match_num).float())


                dev_intent_acc += batch_intent_acc
                dev_slot_acc += batch_slot_acc.sum()
                dev_seq_acc += batch_seq_slot_acc
                count += text.shape[0]

            dev_intent_loss /= count
            dev_slot_loss /= count
            dev_intent_acc /= count
            dev_slot_acc /= count
            dev_seq_acc /= count
            print('test_intent_loss:',dev_intent_loss)
            print('test_intent_acc:', dev_intent_acc)
            print('test_slot_loss:', dev_slot_loss)
            print('test_slot_acc:', dev_slot_acc)
            print('test_utter_acc:', dev_seq_acc)
            self.test_writer.add_scalar('loss/intent_loss', dev_intent_loss, step)
            self.test_writer.add_scalar('loss/slot_loss', dev_slot_loss, step)
            self.test_writer.add_scalar('acc/intent_acc', dev_intent_acc, step)
            self.test_writer.add_scalar('acc/slot_acc', dev_slot_acc, step)
            self.test_writer.add_scalar('acc/seq_acc', dev_seq_acc, step)

        self.model.train()