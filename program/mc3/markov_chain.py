#! Version 3
from markov_dictionary import Markov_Dictionary
from markov_tree import Markov_Tree
import os
import pickle
from matplotlib import pyplot
import seaborn
import numpy

'''
A chain of states that are connected by probabilities.
The state is described by the last n symbols of a word.
The transition to the next state is described by the next symbol.
Therefore, in most cases, it is impossible to remain in the same state.
(Unless the state only has one kind of symbol)
'''


class Markov_Chain():
    # constructor, setting the defaults
    def __init__(self, path, n=0, max_keys=50000):
        self.dict = Markov_Dictionary(1000)
        self.n = n
        self.path = path.rstrip('/')
        self.tree = Markov_Tree(path, n, max_keys)
        ls = sorted(os.listdir(path),
                    key=lambda x: int(x.split('_')[0]))
        if n == 0:
            for i in range(len(ls)-1, 0, -1):
                if os.path.isfile(f'{path}/{ls[i]}'):
                    with open(f'{path}/{ls[i]}', 'rb') as f:
                        if self.get_n(pickle.load(f)):
                            break
        # filename + "-mc3-n" + n
        self.name = f"{self.path.split('/')[-1]}-mc3-n{n}"
        keys = []
        for i in range(len(ls)):
            keys.append(int(ls[i].split('_')[0]))
        # add the very last key
        keys.append(int(ls[-1].split('.')[0].split('_')[1]))
        self.tree.set_keys(keys)

    def get_n(self, mtrx):
        keys = sorted(list(mtrx.keys()),
                      key=lambda x: x[0], reverse=True)
        if len(keys) == 0:
            return False
        self.n = len(self.dict.index_to_state(keys[0][0]))
        print(f'Found Markov Chain with n={self.n}')
        self.dict.depth = self.n
        return True

    def get_probabilities(self, previous, as_nd=False):
        d = self.dict
        if len(previous) > self.n:
            previous = previous[-self.n:]
        if type(previous) == str:
            previous = d.state_to_index(previous)
        if type(previous) == list:
            previous = d.state_to_index(
                ''.join([d.index_to_symbol(x) for x in previous]))
        if as_nd:
            values = self.tree.values(previous, as_nd=True)
            s = values.sum(axis=0)
            if s == 0:
                return values
            return values / s
        else:
            values = self.tree.values(previous)
            s = sum(values)
            if s == 0:
                return [(i, 0) for i in range(len(values))]
            return [(i, values[i]/s) for i in range(len(values))]

    def toheatmap(self, out_path=''):
        print('Generating matrix...')
        if out_path == '':
            p = self.path.split('/')[:-2]
            q = self.path.split('/')[-2:]
            out_path = f"{'/'.join(p)}/visualizations/heatmap-{'-'.join(q)}.png"

        l = self.dict.get_start_index(self.n+1)
        mtrx = numpy.ndarray((l, 95))
        for i in range(l):
            print(f'{i*100/l:.2f}%', end='\r')
            p = self.get_probabilities(i, as_nd=True)
            mtrx[i] = p
        print('Matrix complete, creating heatmap...')
        seaborn.heatmap(mtrx)
        print(f'Saving heatmap to {out_path}...')
        pyplot.savefig(out_path)


# mc = Markov_Chain(
#    '/home/miltfra/projects/Seminararbeit/Data/pwn_0_20.txt-mc/mc3-n2-K100000')
# mc.toheatmap()
