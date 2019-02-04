from queue import Queue
import sys
import os

'''
Takes a file and options as input and outputs
all lines in the file that are compliant with
the options.
Best used with '>>' operator.
'''

# removing unwanted prefixes from file
HEX_START = 'HEX['


class FilePatcher:

    def __init__(self, file, options, output=None):
        self.hex = False
        self.l0 = False
        self.l1 = False
        self.__set_options(options)
        self.file = file
        if output == None:
            self.output = f'{file}-patch/{self}'
        else:
            self.output = output

    def __set_options(self, options):
        for o in options:
            self.__set_option(o[0], o[1])

    def __set_option(self, option, value):
        if option == '-h':
            self.hex = True
        elif option == '-l':
            self.l0 = int(value)
        elif option == '-L':
            self.l1 = int(value)
        else:
            print(f'Unrecognized Option: {option}')

    def run(self):
        with open(self.file) as f:
            size = os.path.getsize(self.file)
            lines_read = 0
            bytes_read = 0
            if not os.path.isdir(f'{self.file}-patch'):
                os.mkdir(f'{self.file}-patch')
            with open(self.output, 'w+') as o:
                l = self.__get_line(f)
                while l != '':
                    if self.__is_compliant(l):
                        o.write(l + '\n')
                    lines_read += 1
                    bytes_read += len(l)
                    print(
                        f'Progress: {bytes_read / size * 100:.2f}%      ', end='\r')
                    l = self.__get_line(f)
        return True

    def __is_compliant(self, l):
        if not self.hex and l[:4] == HEX_START:
            return False
        if self.l0 != False and len(l) < self.l0:
            return False
        if self.l1 != False and len(l) >= self.l1:
            return False
        return True

    def __get_line(self, file):
        return file.readline().strip('\n')

    def __str__(self):
        s = 'fp2p'
        if self.l0 != False:
            s += f'-l{self.l0}'
        if self.l1 != False:
            s += f'-L{self.l1}'
        if self.hex:
            s += '-hex'
        return s


if __name__ == "__main__":
    options = []
    for arg in sys.argv[2:]:
        if arg[0] == '-':
            options.append([arg, ])
        else:
            options[-1].append(arg)
    for o in options:
        o.append(-1)
    fp = FilePatcher(sys.argv[1], options)
    fp.run()
