import models
import data_sets
from mxnet import gluon, nd, init
from matplotlib import pyplot
import numpy
import seaborn
from data_utils import dist2prob, sym2indx, sym2vec, indx2sym
import os
from random import random
import sys

class GRU:

    def __init__(self, net, param_f):
        self.net = net
        self.file = param_f.split('/')[-1]
        self.net.load_parameters(param_f)
        self.reset_word()
        self.reset_hidden(1)

    def next_symbol(self, sym):
        if sym == '\n':
            self.reset_word()
        else:
            self.current_word.append(sym)

    def get_probabilities(self, previous='', as_nd=False):
        if previous != '':
            for sym in previous:
                self.next_symbol(sym)
        data = nd.zeros((1, len(self.current_word), 96))
        for i, s in enumerate(self.current_word):
            data[0, i] = sym2vec(s).reshape(96)
        self.reset_hidden(data.shape[1])
        o, self.current_hidden = self.net(
            data, self.current_hidden)
        if as_nd:
            o = o[0, -1]
            o = nd.softmax(o)
            return o
        return [o[0, -1, i].asscalar() for i in range(96)]

    def reset_word(self):
        self.current_word = ['\n', ]

    def reset_hidden(self, batch_size):
        self.current_hidden = self.net.begin_state(
            batch_size, func=nd.zeros)

    def toheatmap(self, out_path=''):
        if out_path == '':
            out_path = f'/home/miltfra/projects/Seminararbeit/Data/visualizations/heatmap-{".".join(self.file.split(".")[:-1])}.png'
        mtrx = numpy.ndarray((96*96, 96))
        for i in range(96):
            print(f'{i*100/96:.2f}%', end='\r')
            for j in range(96):
                self.next_symbol(indx2sym(i))
                self.next_symbol(indx2sym(j))
                p = self.get_probabilities(as_nd=True)
                mtrx[i*96+j] = p.asnumpy()
        print('Matrix complete, creating heatmap...')
        seaborn.heatmap(mtrx)
        print(f'Saving heatmap to {out_path}...')
        pyplot.savefig(out_path)
        pyplot.close('all')
    def random_word(self, min_length, max_length, start=''):
        w = self.get_candidate(max_length, start=start)
        while not min_length <= len(w) < max_length:
            w = self.get_candidate(max_length)
        return w
    
    def get_candidate(self, max_length, start=''):
        self.reset_word()
        w = []
        for c in start:
            self.current_word.append(c)
            w.append(c)
        c = self.random_symbol()
        w.append(c)
        while (c != 0 or len(w) == 1):
            c = self.random_symbol()
            w.append(c)
            self.next_symbol(indx2sym(c))
            if len(w) == max_length + 1: 
                break
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
    
    def best_symbol(self):
        p = self.get_probabilities(as_nd=True)
        return int(p.argmax(axis=0).asscalar())


def plot(p, n=-1):
    if os.path.isfile(p):
        if n < 0:
            n = int(p.split('-')[-1].split('.')[0])
        gru = GRU(models.gru1(n), p)
        gru.toheatmap(p+'.png')
        return
    all_files = [f for f in os.listdir(
        p) if os.path.isfile(os.path.join(p, f))]
    files = []
    for f in all_files:
        if f.split('.')[-1] == 'params':
            if f.split('-')[0] == 'gru':
                files.append(f)
    for f in files:
        n = int(f.split('-')[-1].split('.')[0])
        gru = GRU(models.gru1(n), p+f)
        gru.toheatmap(p+f+'.png')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        net = GRU(models.gru1(int(sys.argv[2])), sys.argv[1])
        for _ in range(10):
            print(net.random_word(10,11))
    else:
        print("[ERR] Missing argument; Expected: <path> <hidden count>")
