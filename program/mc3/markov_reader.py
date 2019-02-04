#! Version 3
from threading import Thread
import time

'''
Manages the reading of a given file in a separate thread so that there 
is always a buffer left and the main analysis function never has to wait.
'''


class Markov_Reader(Thread):

    # constructor, setting the defaults
    def __init__(self, path, buffer_size=5000000):
        Thread.__init__(self)
        self.path = path
        self.buffer_size = buffer_size
        self.buffer = []
        self.current_symbol = ''

    def run(self):
        with open(self.path) as file:
            while True:
                if len(self.buffer) < self.buffer_size:
                    self.buffer.extend(
                        list(file.read(self.buffer_size)))
                time.sleep(1)

    # returning the next symbol in the buffer in a way that the main
    # analysis can deal with
    def next_symbol(self):
        # if the buffer falls below it's target size, new symbols are
        # read
        if len(self.buffer) == 0:
            # two consecutive \n are interpreted as the end of the file
            # that way the last \n can be registered by the markov chain
            if self.current_symbol == '\n':
                self.current_symbol = ''
            else:
                self.current_symbol = '\n'
        # if the buffer isn't empty, the first symbol is returned
        else:
            self.current_symbol = self.buffer.pop(0)
        return self.current_symbol

    def skip(self, n):
        k = len(self.buffer)
        if n < k:
            self.buffer = self.buffer[n:]
        else:
            self.buffer = []
            self.skip(n - k)
