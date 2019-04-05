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

    #PLOT ACCURACY
    plt.figure(0)
    plt.axes().get_xaxis().set_major_locator(MaxNLocator(integer=True)) #force whole numbers on x axis
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axes().set_yticks(np.arange(0.0,1.0,0.01), minor=True)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min(np.concatenate([dataDict['acc'], dataDict['val_acc']])), 1.)
    plt.plot(dataDict['epoch'], dataDict['acc'], color='red')
    plt.plot(dataDict['epoch'], dataDict['val_acc'], color='blue')
    plt.grid(b=True, which='both')
    plt.legend(('train_acc', 'val_acc'),
               scatterpoints=1,
               loc='center right',
               prop={'size': 16})
    plt.savefig('{}/plot_acc.eps'.format(log_dir), format='eps', bbox_inches='tight',dpi=1200)

    #PLOT LOSS
    plt.figure(1)
    plt.axes().get_xaxis().set_major_locator(MaxNLocator(integer=True)) #force whole numbers on x axis
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(0, max(np.concatenate([dataDict['loss'], dataDict['val_loss']])))
    plt.plot(dataDict['epoch'], dataDict['loss'], color='red')
    plt.plot(dataDict['epoch'], dataDict['val_loss'], color='blue')
    plt.grid(b=True, which='both')
    plt.legend(('train_loss', 'val_loss'),
               scatterpoints=1,
               loc='center right',
               prop={'size': 16})
    plt.savefig('{}/plot_loss.eps'.format(log_dir), format='eps', bbox_inches='tight',dpi=1200)

    #PLOT LEARN RATE
    plt.figure(2)
    plt.axes().get_xaxis().set_major_locator(MaxNLocator(integer=True)) #force whole numbers on x axis
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('lrn_rate', fontsize=15)
    plt.ylim(0, 1.1*max(dataDict['lrn_rate']))
    plt.plot(dataDict['epoch'], dataDict['lrn_rate'], color='red')
    plt.grid(b=True, which='both')
    plt.savefig('{}/plot_lrn_rate.eps'.format(log_dir), format='eps', bbox_inches='tight',dpi=1200)


if __name__ == '__main__':
    plot_training_log('/media/fredrik/WDusbdrive/keras_logdir/2019-04-05_13-25-31.056466/training.log')

