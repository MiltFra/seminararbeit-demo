#! Version 3
'''
Manages the indices and corresponding states of a markov chain.
A dictionary consists of one search tree and a simple list.
The list manages all f(index) = symbol requests while the tree handles
all requests looking for the key of a certain symbol.
This ensures that for every index a symbol can be found easily and 
vice versa.
'''


class Markov_Dictionary:

    def __init__(self, depth):
        self.depth = depth

    def symbol_to_index(self, symbol):
        if symbol == '' or symbol == '\n':
            return 0
        else:
            return ord(symbol) - 31

    def state_to_index(self, state):
        if len(state) > self.depth:
            return None
        s = self.get_start_index(len(state))
        symbols = []
        for c in state:
            symbols.append(self.symbol_to_index(c))
        for i in range(len(symbols)):
            s += (symbols[i]-1) * 95 ** (len(symbols) - i - 1)
        return s

    def get_start_index(self, l):
        a = 0
        for i in range(l):
            a += 95**i
        return a

    def index_to_symbol(self, index):
        if index == 0:
            return '\n'
        else:
            return chr(index + 31)

    def index_to_state(self, index):
        l = 0
        while index >= self.get_start_index(l + 1):
            l += 1
        if l > self.depth:
            return None
        index -= self.get_start_index(l)
        symbols = []
        for i in range(l):
            symbols.append(chr(32 + index // 95 ** (l - i - 1)))
            index %= 95 ** (l - i - 1)
        return ''.join(symbols)
