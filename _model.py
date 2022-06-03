import torch 
from torch import nn
from torch.nn import functional as F


class BaselineModel(nn.Module):
    def __init__(self, seq_len=256, hid_dim=16):
        super().__init__()
        
        self.emb = nn.Embedding(21, hid_dim)
        
        self.p_conv = nn.Sequential(
                nn.Conv1d(hid_dim, hid_dim, kernel_size=3, stride=1, padding=1), 
                nn.LeakyReLU(),
                nn.Conv1d(hid_dim, hid_dim, kernel_size=3, stride=1, padding=1)
                )   
        
        self.attention = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim), 
                nn.LeakyReLU(),
                nn.Linear(hid_dim, 1),
                nn.Sigmoid()
                )
        
        self.pp_conv = nn.Sequential(
                nn.Conv2d(hid_dim * 2, hid_dim, kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(hid_dim, hid_dim, kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                # nn.Conv2d(hid_dim, hid_dim, kernel_size=7, stride=1, padding=3),
                # nn.LeakyReLU(),
                nn.Conv2d(hid_dim, hid_dim, kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(hid_dim, hid_dim, kernel_size=2, stride=2, padding=0)
                )

        readout_dim = int(hid_dim * seq_len * seq_len / 4 / 4 / 4 / 4)
        self.fc_readout = nn.Sequential(
                nn.Linear(readout_dim, hid_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(hid_dim, hid_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(hid_dim, 1)
                )
        
    def forward(self, p1, p2, m1, m2):
        mm = self._get_mm(m1, m2)
        p1 = self.p_conv(self.emb(p1).transpose(1, 2)).transpose(1, 2) * m1.unsqueeze(2)# [b l hd]
        p2 = self.p_conv(self.emb(p2).transpose(1, 2)).transpose(1, 2) * m2.unsqueeze(2)
        pp = self._generate_pp(p1, p2) * mm.unsqueeze(3) # [b l l f*2]
        a = self.attention(pp).repeat(1, 1, 1, pp.size(3)) * mm.unsqueeze(3)
        pp = a * pp
        pp = self.pp_conv(pp.transpose(1, 3)).transpose(1, 3)
        pp = pp.reshape(pp.size(0), -1)
        y = self.fc_readout(pp).squeeze(1)
        y = torch.sigmoid(y)
        return y

    def _generate_pp(self, p1, p2): # [b l f]
        l = p1.size(1)
        p1_repeat = p1.unsqueeze(1).repeat(1, l, 1, 1)
        p2_repeat = p2.unsqueeze(2).repeat(1, 1, l, 1)
        pp = torch.cat([p1_repeat, p2_repeat], dim=3) # [b l l f*2]
        return pp
        
    def _get_mm(self, m1, m2): # [b l]
        l = m1.size(1)
        mm = m1.unsqueeze(1).repeat(1, l, 1) * m2.unsqueeze(2).repeat(1, 1, l)
        return mm

    def get_attention(self, p1, p2, m1, m2):
        mm = self._get_mm(m1, m2)
        p1 = self.p_conv(self.emb(p1).transpose(1, 2)).transpose(1, 2) * m1.unsqueeze(2)# [b l hd]
        p2 = self.p_conv(self.emb(p2).transpose(1, 2)).transpose(1, 2) * m2.unsqueeze(2)
        pp = self._generate_pp(p1, p2) * mm.unsqueeze(3)
        a = self.attention(pp).squeeze(3) * mm
        return a


if __name__ == '__main__':
    import pickle 
    import _dataset as DATASET
    import torch 
    
    sample_data = DATASET.PPIDataset('./data/preprocessed/preprocessed.pkl', max_len=256)[0]

    p1 = torch.Tensor(sample_data['p1']).unsqueeze(0).int()
    p2 = torch.Tensor(sample_data['p2']).unsqueeze(0).int()
    m1 = torch.Tensor(sample_data['m1']).unsqueeze(0)
    m2 = torch.Tensor(sample_data['m2']).unsqueeze(0)
    model = BaselineModel(seq_len=256)

    x = model(p1, p2, m1, m2)
    a = model.get_attention(p1, p2, m1, m2)
    print(a)
    
