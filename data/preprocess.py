import numpy as np 
from tqdm import tqdm

global AA_CHAR
AA_CHAR = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def preprocess_txt(fn, forced_label=None):
    f = open(fn, 'r')
    
    line_list = f.readlines()
    sample_list = []
    for idx, line in enumerate(line_list):
        line = line.strip()
        if len(line.split()) == 3 and not idx == 3:
            protein_1 = line_list[idx + 2].strip()
            protein_2 = line_list[idx + 4].strip()
            sample = {'protein_1':protein_1, 'protein_2':protein_2, 'target':forced_label}

            sample_list.append(sample)

    return sample_list


def tokenize_sample_list(sample_list):
    global AA_CHAR
    tokenized_sample_list = []
    for sample in tqdm(sample_list):
        p1 = sample['protein_1']
        p2 = sample['protein_2']
        p1 = tokenize_string(p1)
        p2 = tokenize_string(p2)
        p1, p2 = np.array(p1), np.array(p2)
        sample.update({'p1':p1, 'p2':p2})
        tokenized_sample_list.append(sample)
    return tokenized_sample_list


def tokenize_string(AA_seq): # 20 + buffer 1
    global AA_CHAR
    s = []
    for char in AA_seq:
        if char in AA_CHAR:
            s.append(AA_CHAR.index(char)) # 0 - 19
        else:
            s.append(len(AA_CHAR))
    return s


if __name__ == '__main__':
    import pickle 

    sample_list = preprocess_txt(fn='./raw/Positive.txt', forced_label=1)
    pos_sample_list = tokenize_sample_list(sample_list)
    print(len(pos_sample_list))
    
    sample_list = preprocess_txt(fn='./raw/Negative.txt', forced_label=0)
    neg_sample_list = tokenize_sample_list(sample_list)

    total_sample_list = pos_sample_list + neg_sample_list
    # pickle.dump(total_sample_list, open('./preprocessed/preprocessed.pkl', 'wb'))
    
    
    # plpt distribution 
    
    from matplotlib import pyplot as plt 
    import seaborn as sns
    from matplotlib import colors

    # plot 1d 
    # """
    len_list = [len(x['p1']) for x in total_sample_list] + [len(x['p2']) for x in total_sample_list]
    len_list = [x for x in len_list if x < 4096]
    sns.distplot(len_list, bins=128)
    plt.title('protein length distribution')
    plt.savefig('./data_len.png')
    plt.clf()
    # """

    # plot 2d distribution 
    """
    len_list_1 = [len(x['p1']) for x in total_sample_list]
    len_list_2 = [len(x['p2']) for x in total_sample_list]
    x, y = [], []
    for idx in range(len(len_list_1)):
        if len_list_1[idx] < 4096 and len_list_2[idx] < 4096:
            x.append(len_list_1[idx])
            y.append(len_list_2[idx])
    
    plt.hist2d(x, y, norm=colors.LogNorm(), cmap=plt.cm.Blues, bins=[128, 128])
    plt.title('protein-protein length distribution (log scale)')
    plt.colorbar()
    plt.savefig('./data_len_2d.png')
    plt.clf()
    """

