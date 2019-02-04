#! Version 3
import os
import pickle
from markov_chain import Markov_Chain
from markov_tree import Markov_Tree
import time


class Markov_Builder:

    def __init__(self, interface, path, n, max_keys=100000):
        self.path = path
        self.n = n
        self.max_keys = max_keys
        self.tree = Markov_Tree(
            f'{path}-mc/{self}', n, max_keys)
        with open(f'{path}-mc/bck/latest.txt') as f:
            self.part_count = int(f.readline().strip('\n'))

    def run(self):
        paths = self._get_bck_files()
        self.tree.reset_dir()
        print(f'[STA] Merging {len(paths)} parts')
        self.current_bck = 0
        self.total_bck = len(paths)
        start = time.time()
        for i in range(self.total_bck):
            self.current_bck = i
            with open(paths[i], 'rb') as f:
                status = self.tree.insert(pickle.load(f))
            if i > 0:
                left = (time.time() - start) / i
            else:
                left = -1
            print(
                f'[STA] Files: {status[0]} / Total Keys: {status[1]} / Last File: {status[2]} / Secs Left: {left}              ', end='\r')

        print(
            f'[STA] Merge completed after {time.time() - start} seconds. Files can be found at {self.tree.path}.')
        return Markov_Chain(self.tree.path, self.n)

    def __str__(self):
        return f'mc3-n{self.n}-K{self.max_keys}'

    def _get_bck_files(self):
        files = []
        for i in range(1, self.part_count+1):
            files.append(self._bck_file(i))
        return files

    def _bck_file(self, x):
        return f'{self.path}-mc/bck/{x}.{self.n}.pkl'
