from matplotlib import pyplot as plt
import torch 
from torch import nn 
from matplotlib import cm

import _model as MODEL 
import _dataset as DATASET


seq_len = 384 
hid_dim = 64 

model_fn = './save/exp_large_done/model_144.pt'
model = MODEL.BaselineModel(seq_len=seq_len, hid_dim=hid_dim)
state_dict = torch.load(model_fn)
new_state_dict = {}

for var in state_dict:
    new_state_dict[var[7:]] = state_dict[var]

model.load_state_dict(new_state_dict)
model.eval()

all_set = DATASET.PPIDataset('./data/preprocessed/preprocessed.pkl', max_len=seq_len)

"""
smallest = 1e10
for idx, data in enumerate(all_set):
    if data['target'] == 1:
        if data['m1'].sum() + data['m2'].sum() < smallest:
            sample_data = data
            smallest = data['m1'].sum() + data['m2'].sum()
            data_idx = idx

print(data_idx)
exit()
"""

import numpy as np

# 3963, 

print(len(all_set))

# sample_data = all_set[3963]
sample_data = all_set[1002]


print(sample_data['protein_1'])
print()
print(sample_data['protein_2'])
print(len(sample_data['protein_1']), len(sample_data['protein_2']))

p1, p2 = sample_data['p1'], sample_data['p2']
m1, m2 = sample_data['m1'], sample_data['m2']
p1, p2, m1, m2 = torch.Tensor(p1), torch.Tensor(p2), torch.Tensor(m1), torch.Tensor(m2)
p1, p2, m1, m2 = p1.unsqueeze(0), p2.unsqueeze(0), m1.unsqueeze(0), m2.unsqueeze(0)
pred_target_p = model(p1.int(), p2.int(), m1, m2)
a = model.get_attention(p1.int(), p2.int(), m1, m2).squeeze(0).detach()

print(a.size())
a = np.array(a)[:int(m2.sum()), :int(m1.sum())]
a = a - a.min()
a = np.power(a, 10)

print(a)
print(pred_target_p)
print(a.shape)

from matplotlib import pyplot as plt 

plt.imshow(a, cmap=cm.Blues)
plt.savefig('./attention.png')
plt.clf()

