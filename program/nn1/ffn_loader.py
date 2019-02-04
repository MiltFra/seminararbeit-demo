import models
import data_sets
from mxnet import gluon, nd, init
from matplotlib import pyplot
import numpy
import seaborn
from data_utils import sym2indx, dist2prob, sym2vec, indx2vec
import sys
import os
from random import random


class FFN:

    def __init__(self, net, param_f):
        self.net = net
        self.file = param_f.split('/')[-1]
        self.net.load_parameters(param_f)
        self.current_word = []

    def get_probabilities(self, previous=None, as_nd=False):
        if previous == None:
            previous = self.current_word
        if len(previous) > 2:
            previous = previous[-2:]
        if len(previous) == 0:
            previous = [0]
        if len(previous) ==1:
            previous = [0, previous[0]]
        in_list = [indx2vec(x) for x in previous]
        in_vec = nd.concat(in_list[0], in_list[1], dim=1)
        v = nd.zeros((1, 192))
        v[0] = in_vec
        data = self.net(v)
        data = dist2prob(data, axis=1)
        # print(data)
        if nd:
            return data[0]
        return [data[0, i].asscalar() for i in range(96)]

    def next_symbol(self, sym):
        i = sym2indx(sym)
        if i == 0:
            self.reset()
        else:
            self.current_word.append(i)
        return self.get_probabilities(as_nd=True)

    def reset(self):
        self.current_word = []

    def toheatmap(self, out_path=''):
        if out_path == '':
            out_path = f'/home/miltfra/projects/Seminararbeit/Data/visualizations/heatmap-{".".join(self.file.split(".")[:-1])}.png'
        mtrx = numpy.ndarray((96*96, 96))
        for i in range(96):
            print(f'{i*100/96:.2f}%', end='\r')
            for j in range(96):
                p = self.get_probabilities(
                    indx2sym(i) + indx2sym(j), as_nd=True)
                mtrx[i*96+j] = p.asnumpy()
        print('Matrix complete, creating heatmap...')
        seaborn.heatmap(mtrx)
        print(f'Saving heatmap to {out_path}...')
        pyplot.savefig(out_path)
        pyplot.close('all')

    def random_word(self, min_length, max_length):
        w = self.get_candidate()
        while not min_length <= len(w) < max_length:
            w = self.get_candidate()
        return w
    
    def get_candidate(self):
        w = []
        c = self.random_symbol()
        w.append(c)
        while c != 0 or len(w) == 0:
            c = self.random_symbol()
            w.append(c)
            self.next_symbol(indx2sym(c))
        s = ''
        for c in w:
            if c != 0:
                s += indx2sym(c)
        return s
    
    def random_symbol(self):
        p = self.get_probabilities(as_nd=True)
        r = random()
        for i in range(p.shape[0]):
            r -= p[i].asscalar()
            if r <= 0:
                return i
        return 0

def indx2sym(indx):
    if indx == 0:
        return '\n'
    return chr(indx + 31)


def print_probabilities(p):
    for i, v in enumerate(p):
        if v > 0:
            if i > 0:
                print(indx2sym(i), ': ', v*100, '%')
            else:
                print(r'\n: ', v*100, '%')


def plot(d):
    all_files = [f for f in os.listdir(
        d) if os.path.isfile(os.path.join(d, f))]
    files = []
    for f in all_files:
        if f.split('.')[-1] == 'params':
            if f.split('-')[0] == 'ff':
                files.append(f)
    for f in files:
        n = int(f.split('-')[-1].split('.')[0])
        ffn = FFN(models.nn1(n), d+f)
        ffn.toheatmap(d+f+'.png')


#a = nd.array([0.1, .2, .3])
# print(dist2prob(a))
if __name__ == "__main__":
    if len(sys.argv) > 2:        
        net = FFN(models.nn1(int(sys.argv[2])), sys.argv[1])
        for _ in range(10):
            print(net.random_word(10, 11))
    else:
        print('[ERR] Missing argument; Expected <path> <hidden count>')