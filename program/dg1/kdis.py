import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

OUT_PATH = '/home/vrelda/projects/Seminararbeit/Program/dg1'
'''
Visualizes the key distribution of a Makrov chain.
The key distribution shows, how many times a any state of
a certain interval/file appeared in the source file.
For better visualization the results are sorted.
'''


def kdis():
    # only y plot needed since it's gonna be sorted and then
    # assigned to indices later
    y = []
    # getting all files from the directory in sys.argv[1]
    files = [f for f in os.listdir(sys.argv[1]) \
        if os.path.isfile(os.path.join(sys.argv[1], f))]
    # counts keys per file
    for f in files:
        # loading the file
        with open(os.path.join(sys.argv[1], f), 'rb') as b:
            dct = pickle.load(b)
        # adding to plot
        y.append(len(dct.keys()))
    # sorting
    y = sorted(y)
    # x-plot, range [0, len(y))
    y_pos = np.arange(len(y))
    # plotting and saving to file
    file = sys.argv[1].split("/")[-1]
    plt.plot(y_pos, y, 'b.')
    plt.title(f'key distribution in {file}')
    plt.xlabel('files ordered by size')
    plt.ylabel('keys per file')
    plt.ylim(0, y[-1]*1.05)
    plt.savefig(f'{OUT_PATH}/kdis-{file}.png')
    plt.close('all')
