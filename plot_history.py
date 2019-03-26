import json
import matplotlib.pyplot as plt
import numpy as np





if __name__ == '__main__':
    with open('Graph/2019-03-25 17:22:54.512982/train_acc.json') as f:
        data = np.array(json.load(f))
        train_acc = np.array(data)[:,2]
        epoch = np.array(data)[:,1]
    with open('Graph/2019-03-25 17:22:54.512982/val_acc.json') as f:
        data = np.array(json.load(f))
        val_acc = np.array(data)[:,2]

    accFig = plt.figure(1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    s_train_acc = plt.plot(epoch, train_acc)
    s_val_acc = plt.plot(epoch, val_acc)
    plt.legend(('train_acc','val_acc'),
               scatterpoints=1,
               loc='center right',
               prop={'size': 16})
    accFig.show()
    #plt.show()
    plt.savefig('Graph/2019-03-25 17:22:54.512982/acc.eps', format='eps')


    with open('Graph/2019-03-25 17:22:54.512982/train_loss.json') as f:
        data = np.array(json.load(f))
        train_loss = np.array(data)[:,2]
        epoch = np.array(data)[:,1]
    with open('Graph/2019-03-25 17:22:54.512982/val_loss.json') as f:
        data = np.array(json.load(f))
        val_loss = np.array(data)[:,2]

    lossFig = plt.figure(2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.plot(epoch, train_loss)
    plt.plot(epoch, val_loss)
    plt.legend(('train_loss','val_loss'),
               scatterpoints=1,
               loc='center right',
               prop={'size': 16})

    lossFig.show()
    #plt.show()
    plt.savefig('Graph/2019-03-25 17:22:54.512982/loss.eps', format='eps')