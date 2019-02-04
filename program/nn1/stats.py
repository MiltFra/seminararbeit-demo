from matplotlib import pyplot as plt
import os
import sys
import numpy
import pickle

def plot(d):
    all_files = [
        f for f in os.listdir(d)
        if os.path.isfile(os.path.join(d, f))
    ]
    files = []
    for f in all_files:
        if f.split('.')[-1] == 'stats':
            files.append(f)
    for f in files:
        plot_f(os.path.join(d, f))

def plot_f(f):
    with open(f, 'rb') as b:
        stats = pickle.load(b)
    epochs = numpy.zeros(len(stats))
    for i in range(len(stats)):
        epochs[i] = i
    loss = numpy.zeros(len(stats))
    train = numpy.zeros(len(stats))
    valid = numpy.zeros(len(stats))
    for i in range(len(stats)):
        loss[i] = stats[i][1]
        train[i] = stats[i][2]
        valid[i] = stats[i][3]
    max_loss = numpy.max(loss, axis=0)
    min_loss = numpy.min(loss, axis=0)
    loss = (loss -
            min_loss) / (max_loss - min_loss)
    plt.plot(epochs, loss, 'b--')
    plt.plot(epochs, train, 'b-')
    plt.plot(epochs, valid, 'r-')
    plt.title(
        f'loss, train and valid of {f.split("/")[-1]}'
    )
    plt.savefig(f + '.png')
    plt.close('all')