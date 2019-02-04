#! Version 3
import os
import pickle
from collections import Counter
import queue
import numpy


class Markov_Tree:

    def __init__(self, path, n, max_keys=50000):
        self.path = path
        self.max_keys = max_keys
        self.c = self.start_b(n)
        self.root = Markov_Node(self, 0, self.c // 2, self.c)
        self.count = 0
        self.inserts = 0

    def reset_dir(self):
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        else:
            for f in os.listdir(self.path):
                path = os.path.join(self.path, f)
                try:
                    if os.path.isfile(path):
                        os.unlink(path)
                except Exception as e:
                    print(e)

    def set_keys(self, keys):
        self.root.set_keys(keys)

    def start_b(self, n):
        b = 0
        for i in range(n + 1):
            b += 95**i
        return b

    def insert(self, dct):
        fulldict = dict(dct)
        node = self.root
        stack = queue.LifoQueue()
        stack.put((node, dct))
        i = 1
        while i > 0:
            node, dct = stack.get()
            if isinstance(node, Markov_Leaf):
                node.insert(dct)
                i -= 1
            else:
                dct0, dct1 = self.split_dictionary(
                    node.keys[1], dct)
                if node.left == None:
                    node.left = Markov_Leaf(
                        node.tree, node, node.keys[0], node.keys[1])
                if node.right == None:
                    node.right = Markov_Leaf(
                        node.tree, node, node.keys[1], node.keys[2])
                stack.put((node.left, dct0))
                stack.put((node.right, dct1))
                i += 1
        self.inserts += 1
        return (self.inserts, self.count, len(fulldict))
    # Binary search for the first index with a value that is >= key

    def split_dictionary(self, key, dct):
        keys = sorted(list(dct.keys()))
        a = 0
        c = len(keys)
        b = (c - a) // 2 + a
        while not self.is_split_index(b, key, keys):
            if keys[b][0] < key:
                a = b
            else:
                c = b
            b = (c - a) // 2 + a
        dct0 = {}
        dct1 = {}
        for i in range(b):
            dct0[keys[i]] = dct[keys[i]]
        for i in range(b, len(keys)):
            dct1[keys[i]] = dct[keys[i]]
        return dct0, dct1

    def is_split_index(self, s, k, keys):
        if s == len(keys) and keys[s-1][0] < k:
            return True
        if s == 0 and keys[s][0] >= k:
            return True
        return keys[s-1][0] < k <= keys[s][0]

    def values(self, state, as_nd=False):
        return self.root.values(state, as_nd=as_nd)

    def __str__(self):
        return f'[{str(self.root)}]'


class Markov_Node:

    def __init__(self, tree, a, b, c):
        self.tree = tree
        self.keys = (a, b, c)
        self.left = None
        self.right = None

    def set_children(self, left, right):
        self.left = left
        self.right = right

    def set_keys(self, keys):
        # all values are compared to keys[len(keys) // 2]
        self.keys = (keys[0], keys[len(keys) // 2], keys[-1])
        # keys[len(keys) // 2] has to be in both lists as the upper/lower limit
        keys0 = keys[:len(keys) // 2 + 1]
        keys1 = keys[len(keys) // 2:]
        if len(keys0) == 2:
            self.left = Markov_Leaf(
                self.tree, self, keys0[0], keys0[1])
        else:
            self.left = Markov_Node(self.tree, 0, 0, 0)
            self.left.set_keys(keys0)
        if len(keys1) == 2:
            self.right = Markov_Leaf(
                self.tree, self, keys1[0], keys1[1])
        else:
            self.right = Markov_Node(self.tree, 0, 0, 0)
            self.right.set_keys(keys1)

    def insert(self, dct):
        keys = sorted(list(dct.keys()), key=lambda x: x[0])
        dct0 = {}
        dct1 = {}
        for k in keys:
            if k[0] < self.keys[1]:
                dct0[k] = dct.get(k)
            else:
                dct1[k] = dct.get(k)
        if self.left == None:
            self.left = Markov_Leaf(
                self.tree, self, self.keys[0], self.keys[1])
        if self.right == None:
            self.right = Markov_Leaf(
                self.tree, self, self.keys[1], self.keys[2])
        self.left = self.left.insert(dct0)
        self.right = self.right.insert(dct1)
        return self

    def values(self, state, as_nd=False):
        if state < self.keys[1]:
            return self.left.values(state, as_nd=as_nd)
        else:
            return self.right.values(state, as_nd=as_nd)

    def __str__(self):
        return f'{str(self.left)}|{str(self.right)}'


class Markov_Leaf:

    def __init__(self, tree, node, a, b):
        self.tree = tree
        self.node = node
        self.keys = (a, b)
        self.path = f'{tree.path}/{a}_{b}.pkl'
        if not os.path.isfile(self.path):
            with open(self.path, 'wb') as f:
                pickle.dump({}, f, protocol=-1)

    def insert(self, dct1):
        with open(self.path, 'rb') as f:
            dct0 = pickle.load(f)
        self.tree.count -= len(dct0)
        dct0 = Counter(dct0) + Counter(dct1)
        self.tree.count += len(dct0)
        if len(list(dct0.keys())) > self.tree.max_keys:
            return self.split(dct0)
        with open(self.path, 'wb') as f:
            pickle.dump(dct0, f, protocol=-1)
        return self

    def values(self, state, as_nd=False):
        with open(self.path, 'rb') as f:
            dct = pickle.load(f)
        if as_nd:
            v = numpy.ndarray((95))
            for i in range(95):
                v[i] = dct.get((state, i), 0)
            return v
        else:
            v = []
            for i in range(95):
                v.append(dct.get((state, i), 0))
            return v

    def split(self, dct):
        keys = sorted(list(dct.keys()), key=lambda x: x[0])
        a = self.keys[0]
        b = keys[len(keys)//2][0]
        c = self.keys[1]
        node = Markov_Node(self.tree, a, b, c)
        node.left = Markov_Leaf(self.tree, node, a, b)
        node.right = Markov_Leaf(self.tree, node, b, c)
        node.insert(dct)
        os.remove(self.path)
        if self.node.left == self:
            self.node.left = node
        else:
            self.node.right = node
        return node

    def __str__(self):
        return f'{self.keys[0]}-{self.keys[1]}'
