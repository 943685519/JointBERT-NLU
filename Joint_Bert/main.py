from trainer import Trainer
from config import config
from model import nluModel
from dataset import nluDataset

config = config(use_crf=False)
model = nluModel(config)
train_dataset = nluDataset(config,config.train_file)
test_dataset = nluDataset(config,config.test_file)

# trainer = Trainer(config=config,model=model,train_dataset=train_dataset,valid_dataset=test_dataset)
trainer = Trainer(config=config,model=model,train_dataset=train_dataset,valid_dataset=test_dataset)
for i in range(31):
    trainer.train(i)