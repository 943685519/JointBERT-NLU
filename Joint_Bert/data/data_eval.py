from collections import defaultdict

intent_label_num = defaultdict(int)
slot_label_num = defaultdict(int)
intent_label_num_test = defaultdict(int)
slot_label_num_test = defaultdict(int)
with open('train.txt', 'r', encoding='utf8') as f:
    lines = [line.split('\t') for line in f.readlines()]
    for text, intent, slot in lines:
        intent_label_num[intent]+=1
        slot_list = [i for i in slot.split()]
        for i in slot_list:
            slot_label_num[i]+=1

with open('test.txt', 'r', encoding='utf8') as f:
    lines = [line.split('\t') for line in f.readlines()]
    for text, intent, slot in lines:
        intent_label_num_test[intent]+=1
        slot_list = [i for i in slot.split()]
        for i in slot_list:
            slot_label_num_test[i]+=1

with open('intent_label_eval.txt','w', encoding='utf8') as f:
    for k,v in intent_label_num.items():
        f.write(str(k)+'\t'+str(v)+'\n')

with open('slot_label_eval.txt','w', encoding='utf8') as f:
    for k,v in slot_label_num.items():
        f.write(str(k)+'\t'+str(v)+'\n')

with open('intent_label_eval_test.txt','w', encoding='utf8') as f:
    for k,v in intent_label_num_test.items():
        f.write(str(k)+'\t'+str(v)+'\n')

with open('slot_label_eval_test.txt','w', encoding='utf8') as f:
    for k,v in slot_label_num_test.items():
        f.write(str(k)+'\t'+str(v)+'\n')