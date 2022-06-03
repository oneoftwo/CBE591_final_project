import torch 
import numpy as np
import pickle

class PPIDataset(torch.utils.data.Dataset):
    def __init__(self, sample_list_fn, max_len=1000):
        super().__init__()
        self.sample_list = pickle.load(open(sample_list_fn, 'rb'))
        self.max_len = max_len
        self._sanitize_sample_list()
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        sample['m1'] = self._generate_mask(sample['p1'])
        sample['m2'] = self._generate_mask(sample['p2'])
        sample['p1'] = self._pad_aa_seq(sample['p1'])
        sample['p2'] = self._pad_aa_seq(sample['p2'])
        return sample
    
    def _sanitize_sample_list(self):
        new_sample_list = []
        for sample in self.sample_list:
            if len(sample['p1']) < self.max_len and len(sample['p2']) < self.max_len:
                new_sample_list.append(sample)
        self.sample_list = new_sample_list

    def _generate_mask(self, aa_seq):
        l = len(aa_seq)
        m = np.array([1 for _ in range(l)] + [0 for _ in range(self.max_len - l)])
        return m

    def _pad_aa_seq(self, aa_seq):
        to_pad = self.max_len - len(aa_seq)
        aa_seq = np.pad(aa_seq, (0, to_pad))
        return aa_seq


    
if __name__ == '__main__':
    ppi_dataset = PPIDataset('../FINAL/data/preprocessed/preprocessed.pkl')
    print(len(ppi_dataset))
    print(ppi_dataset[2]['m1'])
    print(ppi_dataset[2]['p1'])
    

