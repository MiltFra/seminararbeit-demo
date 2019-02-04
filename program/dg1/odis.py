import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

OUT_PATH = '/home/miltfra/projects/Seminararbeit/Program/dg1'

'''
Visualizes the occurence distribution of a Markov chain.
The occurence distribution shows, how many times
any state of a certain interval appeared in the source file.
For better visualization the result is in log_10.
'''


def odis():
    # creating x and y for plot
    y = []
    x = []
    # getting all files from the directory in sys.argv[1]
    files = [f for f in os.listdir(sys.argv[1]) \
        if os.path.isfile(os.path.join(sys.argv[1], f))]
    # counts occurrences (hits) for every file
    for f in files:
        # loading file to read from
        with open(os.path.join(sys.argv[1], f), 'rb') as b:
            dct = pickle.load(b)
        f = f.split('.')[0].split('_')
        # sum of all the values in the file
        hits = 0
        for k in dct.keys():
            hits += dct[k]
        # adding to plot
        y.append(math.log(hits, 10))
        x.append(int(f[0]))
    # plotting and saving to file
    file = sys.argv[1].split("/")[-1]
    plt.plot(x, y, 'g.')
    plt.title(f'occurence density in {file}')
    plt.xlabel('intervall start')
    plt.ylabel('log10 occurences per file')
    plt.savefig(f'{OUT_PATH}/odis-{file}.png')
    plt.close('all')
