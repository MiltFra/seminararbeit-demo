#! Version 3
from markov_reader import Markov_Reader
from markov_dictionary import Markov_Dictionary
from queue import Queue
import queue
from threading import Thread
from markov_queue import Markov_Queue
from markov_chain import Markov_Chain
import numpy as np
from copy import deepcopy
import pickle
import os
import time


class Markov_Analysis():

    def __init__(self, path='', n0=1, n1=5, buffer_size=50000, skip=0, backup_bytes=5000000, resume=False, worker_count=1):
        self.path = path
        if resume:
            resume = self.resume()
        if not resume:
            self.size = os.path.getsize(path)
            self.backup_bytes = backup_bytes
            self.backups_done = 0
            self.buffer_size = buffer_size
            self.complete = False
            self.count = skip
            self.n0 = n0
            self.n1 = n1
            self.occs = []
            self.dict = Markov_Dictionary(n1)
            for _ in range(n0, n1):
                self.occs.append({})
        self.word_queue = Queue()
        self.reader = Markov_Analysis_Reader(self)
        self.workers = [Markov_Analysis_Worker(
            self) for i in range(worker_count)]
        self.run()

    def run(self):
        print(
            f'Analysis({self.path}, {self.n0}, {self.n1}, {self.backup_bytes}) - {self.size} Bytes')
        start = time.time()
        last = -1
        rep = 0
        while self.count < self.size and rep <= 100:
            if self.count > (self.backups_done + 1) * self.backup_bytes > 0:
                self.backup()
            total_progress = self.count * 100 / self.size
            if total_progress == 0:
                t_remaining = -1
            else:
                t_remaining = (time.time()-start) * \
                    (100-total_progress)/total_progress
            print('Progress: {:.6f}/{:.6f} GB ({:.6f}%); Complete in {:.0f} Seconds    '.format(
                self.count/1000000000, self.size/1000000000, total_progress, t_remaining), end='\r')
            if self.count == last:
                rep += 1
            else:
                last = self.count
                rep = 0
            time.sleep(2)
        self.backup()
        self.complete = True 
        print(
            f'Analysis completed! Part files can be found at: {self.path}-mc/bck/')

    def resume(self):
        path = '{}-mc/bck'.format(self.path)
        if not os.path.isfile('{}/latest.txt'.format(path)):
            return False
        with open('{}/latest.txt'.format(path), 'r') as f:
            self.backups_done, self.n0, self.n1, self.count, self.backup_bytes, self.buffer_size = (
                int(f.readline().rstrip('\n')) for i in range(6))
        self.size = os.path.getsize(self.path)
        self.dict = Markov_Dictionary(self.n1)
        self.occs = [{} for i in range(self.n0, self.n1)]
        return True

    def backup(self):
        path = '{}-mc/bck'.format(self.path)
        if not os.path.isdir(path):
            os.makedirs(path)
        occs = self.occs
        self.occs = [{} for i in range(self.n0, self.n1)]
        for i in range(len(occs)):
            occ_path = '{}/{}.{}.pkl'.format(
                path, self.backups_done + 1, i + self.n0)
            print(occ_path)
            with open(occ_path, 'wb') as f:
                pickle.dump(occs[i], f,
                            pickle.HIGHEST_PROTOCOL)
        with open('{}/latest.txt'.format(path), 'w+') as f:
            f.write(str(self.backups_done + 1) + '\n')
            f.write(str(self.n0) + '\n')
            f.write(str(self.n1) + '\n')
            f.write(str(self.count) + '\n')
            f.write(str(self.backup_bytes) + '\n')
            f.write(str(self.buffer_size) + '\n')
        self.backups_done += 1


class Markov_Analysis_Reader(Thread):

    def __init__(self, analysis, skip=0):
        Thread.__init__(self)
        self.path = analysis.path
        self.buffer = analysis.buffer_size
        self.analysis = analysis
        self.skip = skip
        self.start()

    def run(self):
        print("[STA] Spawning Analysis Reader")
        with open(self.path) as file:
            while self.skip > 0:
                file.readline()
                self.skip -= 1
            line = file.readline()
            while line != '' and not self.analysis.complete:
                while self.analysis.word_queue.qsize() < self.buffer:
                    self.analysis.word_queue.put(line)
                    line = file.readline()
        print("[STA] Killing Analysis Reader")


class Markov_Analysis_Worker(Thread):

    def __init__(self, analysis):
        Thread.__init__(self)
        self.n0 = analysis.n0
        self.n1 = analysis.n1
        self.analysis = analysis
        self.dict = Markov_Dictionary(self.n1)
        self.start()

    def run(self):
        print('[STA] Spawning Analysis Worker')
        while not self.analysis.complete:
            try:
                word = self.analysis.word_queue.get(timeout=1)
            except queue.Empty:
                pass
            if word != None:
                self.analyze_word(word)
        print('[STA] Killing Analysis Worker')

    def analyze_word(self, word):
        for i in range(self.n0, self.n1):
            q = Markov_Queue(i)
            for j in range(len(word)):
                c = self.analysis.dict.symbol_to_index(
                    word[j])
                s = self.analysis.dict.state_to_index(
                    q.get_text())
                dct = self.analysis.occs[i-self.n0]
                dct[(s, c)] = dct.get((s, c), 0) + 1
                q.append_symbol(word[j])
        self.analysis.count += len(word)
