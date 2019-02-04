#! Version 3
from markov_queue import Markov_Queue
from random import random

'''
Creates a random text from a given markov chain.
If a length is given, the process is repeated until a candidate is found that has
the right length.
'''


class Markov_Word:

    # constructor, setting the defaults
    def __init__(self, chain, min_length=0, max_length=10000):
        self.chain = chain
        self.min_length = min_length
        self.max_length = max_length
        self.text = ''
        self.relative_average_score = 0
        self.absolute_average_score = 0
        self.absolute_total_score = 0

    # returns a random word
    def random(self):
        # a candidates is generated
        i = 0
        canditate = self.get_candidate()
        # the length is checked, if it's fitting the candidate is returned
        if self.max_length == self.min_length:
            return canditate
        while not self.min_length <= len(canditate) < self.max_length:
            i += 1
            print(f'[STA] Candidates discarded: {i}', end='\r')
            canditate = self.get_candidate()
        return canditate

    # generates a random word from the given markov chain
    def get_candidate(self):
        self.text = ''
        self.absolute_total_score = 0
        last = self.random_symbol()
        queue = Markov_Queue(self.chain.n).append_symbol(last)
        while last != '':
            last = self.random_symbol(queue.get_text())
            queue.append_symbol(last)
            self.text += last
        return self.text

    def update_score(self, previous, probabilities, i):
        if self.text == '':
            self.absolute_average_score = 0
            self.absolute_total_score = 0
            self.relative_average_score = 0
        else:
            self.absolute_total_score += self.get_absolute_score(
                previous, probabilities, i)
            self.absolute_average_score = self.absolute_total_score / \
                len(self.text)
            self.relative_average_score = 1 - \
                self.absolute_average_score / 95

    # gets a random symbol based on the previous state of the chain
    def random_symbol(self, previous=''):
        # gets the probabilities from the chain
        probabilities = self.chain.get_probabilities(previous)
        probabilities = sorted(
            probabilities, key=lambda x: x[1], reverse=True)
        # generates a random in [0, 1)
        r = random()
        # substracts the probabilities until r is smaller than the next one
        # that way all the symbols are distributed with their probabilities
        for i in range(len(probabilities)):
            if r < probabilities[i][1]:
                s = self.chain.dict.index_to_symbol(
                    probabilities[i][0])
                if s == '\n':
                    return ''
                else:
                    self.update_score(previous, probabilities, i)
                    return s
            r -= probabilities[i][1]
        return ''

    def get_absolute_score(self, previous, probabilities, index):
        probs2 = []
        for i in range(len(probabilities)):
            probs2.append((probabilities[i][1], i))
        probs2.sort(key=lambda p: p[0], reverse=True)
        for i in range(len(probs2)):
            if probs2[i][1] == index:
                return i
