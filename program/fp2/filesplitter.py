import os
import sys

'''
Takes a file and options as input and outputs all lines
in the file that are compliant with the options.
Best used with '>>' operator.
'''


class FileSplitter:

    def __init__(self, file, options, output=None):
        self.partsize = False
        self.partcount = False
        self.file = file
        if output == None:
            self.output = f'{file}-{self}'
        else:
            self.output = output
        self.__set_options(options)

    def __set_options(self, options):
        if len(options) == 0:
            self.partcount = 2
        else:
            if options[-1][0] == '-s':
                self.partsize = int(options[-1][1])
            elif options[-1][0] == '-c':
                self.partcount = int(options[-1][1])
            else:
                print(f'Unrecognized option: {options[-1]}')
                return False
        if self.partsize == False:
            self.partsize = os.path.getsize(
                self.file) // self.partcount + 1
        else:
            self.partcount = os.path.getsize(
                self.file) // self.partsize + 1

    def __get_line(self, file):
        return file.readline()

    def run(self):
        with open(self.file) as f:
            size = os.path.getsize(self.file)
            total_bytes_read = 0
            splits = 0
            self.output = f'{self.file}-{self}'
            if not os.path.isdir(self.output):
                os.mkdir(self.output)
            l = self.__get_line(f)
            while l != "":
                bytes_read = 0
                with open(f'{self.output}/part{splits}', 'w') as o:
                    while l != "" and bytes_read < self.partsize:
                        bytes_read += len(l)
                        total_bytes_read += len(l)
                        o.write(l)
                        print(
                            f"Progress: {total_bytes_read / size * 100:.2f}%    ", end='\r')
                        l = self.__get_line(f)
                    splits += 1

    def __str__(self):
        return f'fp2s-s{self.partsize}-c{self.partcount}'


if __name__ == "__main__":
    options = []
    for arg in sys.argv[2:]:
        if arg[0] == '-':
            options.append([arg, ])
        else:
            options[-1].append(arg)
    for o in options:
        o.append(-1)
    fs = FileSplitter(sys.argv[1], options)
    fs.run()
