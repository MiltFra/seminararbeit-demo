# library
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys
import math
import pickle
from collections import Counter
import matplotlib.pyplot as plt
OUT_PATH = '/home/miltfra/projects/Seminararbeit/Program/dg1'
def heatmap():
    fdct = Counter()
    files = [f for f in os.listdir(sys.argv[1]) if os.path.isfile(os.path.join(sys.argv[1], f))]
    for f in files:
        with open(os.path.join(sys.argv[1], f), 'rb') as b:
            fdct += pickle.load(b)
        f = f.split('.')[0].split('_')
    # Create a dataset (fake)
    mtrx = []
    for i in range(95):
        mtrx.append([])
        for j in range(95):
            mtrx[i].append(math.log10(fdct.get((i, j), 1)))
    df = pd.DataFrame(mtrx)
 

    file =  sys.argv[1].split("/")[-1]
    # Default heatmap: just a visualization of this square matrix
    sns.heatmap(df)
    plt.savefig(f'{OUT_PATH}/heatmap-{file}')

heatmap()