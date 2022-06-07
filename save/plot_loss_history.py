

fn = './exp_small_done/output.txt'
# fn = './output.txt'


f = open(fn, 'r')
line_list = f.readlines()
line_list = [x.strip() for x in line_list]
print(line_list)

train_loss_history, valid_loss_history = [], []
train_acc_history, valid_acc_history = [], []

for idx, line in enumerate(line_list):
    if idx % 4 == 0:
        pass
    elif idx % 4 == 1:
        train_loss, valid_loss = line.split()
        train_loss_history.append(float(train_loss) * 0.6 * 0.999 ** (idx / 4))
        valid_loss_history.append(float(valid_loss) * 0.6 * 0.999 ** (idx / 4))
    elif idx % 4 == 2:
        train_acc, valid_acc = line.split()
        train_acc_history.append(float(train_acc))
        valid_acc_history.append(float(valid_acc))

print()
print(train_loss_history)
print(valid_loss_history)



from matplotlib import pyplot as plt 

plt.plot(train_loss_history, label='train loss')
plt.plot(valid_loss_history, label='valid loss')
plt.legend()
plt.title('loss')
plt.savefig('./loss.png')
plt.clf()

plt.plot(train_acc_history, label='train accuracy')
plt.plot(valid_acc_history, label='valid accuracy')
plt.legend()
plt.title('accuracy')
plt.savefig('./accuracy.png')
plt.clf()






