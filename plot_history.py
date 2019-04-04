import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os

def plot_training_log(log_path):
    log_dir = os.path.dirname(log_path)
    print(log_dir)
    with open(log_path) as f:
        rawData = []
        for line in f.readlines():
            rawData.append(line.strip('\n').split(','))
    keys = np.array(rawData[0])
    print('Available data: {}'.format(keys))
    values = np.array(rawData[1:]).astype(float)
    dataDict = {}
    for i,key in enumerate(keys):
        dataDict[key] = values[:,i]
    dataDict['epoch'] = dataDict['epoch'].astype('int')

    accFig = plt.figure(0)
    plt.axes().get_xaxis().set_major_locator(MaxNLocator(integer=True)) #force whole numbers on x axis
    plt.xticks(fontsize=12)
    plt.axes().set_xticks(np.arange(0, max(dataDict['epoch'])+1, 1), minor=True)
    plt.yticks(fontsize=12)
    plt.axes().set_yticks(np.arange(0.0,1.0,0.01), minor=True)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min(np.concatenate([dataDict['acc'], dataDict['val_acc']])), 1.)
    plt.plot(dataDict['epoch'], dataDict['acc'])
    plt.plot(dataDict['epoch'], dataDict['val_acc'])
    plt.grid(b=True, which='both')
    plt.legend(('train_acc', 'val_acc'),
               scatterpoints=1,
               loc='center right',
               prop={'size': 16})
    #accFig.show()
    # plt.show()
    plt.savefig('{}/plot_acc.eps'.format(log_dir), format='eps')

    lossFig = plt.figure(1)
    plt.axes().get_xaxis().set_major_locator(MaxNLocator(integer=True)) #force whole numbers on x axis
    plt.xticks(fontsize=12)
    plt.axes().set_xticks(np.arange(0, max(dataDict['epoch']) + 1, 1), minor=True)
    plt.yticks(fontsize=12)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(0, max(np.concatenate([dataDict['loss'], dataDict['val_loss']])))
    plt.plot(dataDict['epoch'], dataDict['loss'])
    plt.plot(dataDict['epoch'], dataDict['val_loss'])
    plt.grid(b=True, which='both')
    plt.legend(('train_loss', 'val_loss'),
               scatterpoints=1,
               loc='center right',
               prop={'size': 16})
    #accFig.show()
    # plt.show()
    plt.savefig('{}/plot_loss.eps'.format(log_dir), format='eps')



if __name__ == '__main__':
    plot_training_log('/media/fredrik/WDusbdrive/keras_logdir/2019-04-04_10-58-56.785302/training.log')

