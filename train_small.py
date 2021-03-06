import torch
from torch import nn
from tqdm import tqdm


def process(model, data_loader, args, optimizer=None):
    if optimizer != None:
        model.train()
    else:
        model.eval()
    criterion = nn.BCELoss(reduction='sum')

    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(data_loader):
        p1, p2 = batch['p1'].cuda(), batch['p2'].cuda()
        m1, m2 = batch['m1'].cuda(), batch['m2'].cuda()
        pred_target_p = model(p1, p2, m1, m2).cpu()
        
        loss = criterion(pred_target_p, batch['target'].float())

        if optimizer != None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        pred_target = (pred_target_p > 0.5)
        total_loss += loss.item()
        correct += (pred_target == batch['target']).sum().item()
        total += p1.size(0)
    
    avg_loss = total_loss / total 
    acc = correct / total

    return model, avg_loss, acc


if __name__ == '__main__':
    
    seq_len = 256
    hid_dim = 64
    bs = 32
    lr = 1e-5

    import _model as MODEL
    from torch import optim
    from torch.utils.data import DataLoader
    import _dataset as DATASET 
    
    model = MODEL.BaselineModel(seq_len=seq_len, hid_dim=hid_dim)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    
    train_dataset = DATASET.PPIDataset('./data/preprocessed/preprocessed.pkl', max_len=seq_len)
    n = len(train_dataset)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [int(n*0.8), n - int(n*0.8)])
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)

    for epoch_idx in range(1, 1001):
        model.cuda()
        model, train_loss, train_acc = process(model, train_loader, None, optimizer=optimizer)
        model, valid_loss, valid_acc = process(model, valid_loader, None)
        
        print(epoch_idx)
        print(train_loss, valid_loss)
        print(train_acc, valid_acc)
        print()
        
        model.cpu()
        torch.save(model.cpu().state_dict(), f'./save/exp_small/model_{epoch_idx}.pt')

