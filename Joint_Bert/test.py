import torch
torch.manual_seed(13)

x = torch.tensor([[0,1,2,3,4,0,34,0,0,0,0,0],
                  [0,1,2,34,4,3,2,5,6,4,0,0],
                  [0,0,24,2,4,56,7,8,0,0,0,0]])

y = torch.tensor([[0,1,2,3,4,0,34,0,0,0,0,0],
                  [0,1,2,34,4,3,2,5,6,4,0,0],
                  [0,0,24,2,4,56,7,8,0,0,0,0]])

mask = 1 - torch.eq(x,0).float()
out = (x==y).float()
slot_num = torch.sum(mask,dim=-1)

match_num = torch.sum(out*mask,dim=-1)


print('slot_num',slot_num)
print('match_num',match_num)
print(torch.sum((slot_num==match_num).float()))