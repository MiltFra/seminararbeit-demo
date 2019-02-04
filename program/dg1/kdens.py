import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

OUT_PATH = '/home/vrelda/projects/Seminararbeit/Program/dg1'
'''
Visualizes the key density of a Markov chain.
The key density shows the key distribution devided by the
maximum possible value for the interface. In other words:
the fraction of the maximum possible entries in the interval
that has actually been used.
'''

def kdens():
    # creating x and y for plot
    y = []
    x = []
    # getting all files from the directory in sys.argv[1]
    files = [f for f in os.listdir(sys.argv[1]) \
        if os.path.isfile(os.path.join(sys.argv[1], f))]
    # counts keys per file and divides them by the maximum
    for f in files:
        # loading the file
        with open(os.path.join(sys.argv[1], f), 'rb') as b:
            dct = pickle.load(b)
        # getting interval from file name
        f = f.split('.')[0].split('_')
        # calculating interval length
        delta = int(f[1]) - int(f[0])
        # dividing key count by maximum value
        y.append(len(dct.keys())/(95*delta))
        # interval start as x
        x.append(int(f[0]))
    # plotting and saving to file
    file = sys.argv[1].split("/")[-1]
    plt.plot(x, y, 'r.')
    plt.title(f'key density in {file}')
    plt.xlabel('intervall start')
    plt.ylabel('unique states per states in the intervall')
    plt.ylim(0, max(y) * 1.05)
    plt.savefig(f'{OUT_PATH}/kdens-{file}.png')
    plt.close()
