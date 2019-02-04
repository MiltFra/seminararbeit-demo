import os
import sys
import pickle

START_TOKEN = 0
END_TOKEN = 96
'''
Dumps a tokenized version of the input string into a
file. Every line starts with START_TOKEN and ends with
END_TOKEN.
'''


class FileTokenizer:

    def __init__(self, file, silent=True, output=None):
        if output == None:
            self.output = f'{file}-{self}'
        else:
            self.output = output
        self.__data = []
        self.file = file
        self.silent = silent

    def run(self):
        with open(self.file) as f:
            size = os.path.getsize(self.file)
            bytes_read = 0
            l = self.__get_line(f)
            while l != '':
                self.__tokenize(l)
                bytes_read += len(l)
                print(
                    f'Progress: {bytes_read / size * 100:.2f}%; {l}  ', end='\r')
                l = self.__get_line(f)
        with open(self.output, 'wb') as f:
            pickle.dump(self.__data, f, protocol=-1)
        if not self.silent:
            self.__print_data()

    def __get_line(self, f):
        return f.readline().strip('\n')

    def __tokenize(self, w):
        tokens = [START_TOKEN, ]
        for symbol in w:
            tokens.append(ord(symbol) - 32)
        tokens.append(END_TOKEN)
        self.__data.append(tokens)

    def __print_data(self):
        s = []
        for w in self.__data:
            s = ''
            for t in w:
                s += f'{t:3}'
            print(s)

    def __str__(self):
        return f'fp2t'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ft = FileTokenizer(sys.argv[1], silent=False)
        ft.run()
