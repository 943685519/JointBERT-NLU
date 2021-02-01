(Unofficial) Pytorch implementation of `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68875755-b2f92900-0746-11ea-8819-401d60e4185f.png" />  
</p>

- Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)
- batch_loss = batch_intent_loss + batch_slot_loss
- **If you want to use CRF layer, set 'use-crf'=True in main.py**
- seq_acc is the whole sentence accuracy when doing slot filling

In order to run this code, please download bert model first.
