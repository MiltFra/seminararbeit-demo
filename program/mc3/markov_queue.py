#! Version 3
'''
Saves the last n symbols that were given.
If with a given symbol the lenght is exceeded, the first one is removed.
'''


class Markov_Queue:

    def __init__(self, n):
        self.array = ['' for i in range(n)]
        self.n = n

    def get_text(self):
        return ''.join(self.array)

    def append_symbol(self, sym):
        if sym == '\n':
            self.__init__(self.n)
            return self
        self.array.append(sym)
        self.array.pop(0)
        return self
