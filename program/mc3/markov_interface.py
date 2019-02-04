#! Version 3
from markov_chain import Markov_Chain
from markov_word import Markov_Word
from markov_reader import Markov_Reader
from threading import Thread
from markov_analysis import Markov_Analysis
from markov_builder import Markov_Builder
import os
import sys


def clear(): return os.system('cls')


class Markov_Interface():

    def __init__(self):
        self.active_chain = None
        self.random_output_file = None

    def process(self, command):
        tokens = command.split(' ')
        if tokens[0] == 'analyze':
            self.cmd_analyze(tokens)
        elif tokens[0] == 'resume':
            self.cmd_resume(tokens)
        elif tokens[0] == 'read':
            self.cmd_read(tokens)
        elif tokens[0] == 'random':
            self.cmd_random(tokens)
        elif tokens[0] == 'build':
            self.cmd_build(tokens)
        elif tokens[0] == 'exit':
            return False
        elif tokens[0] == 'clear':
            clear()
        else:
            print(f'Unknown command {tokens[0]}')
        return True

    def cmd_analyze(self, tokens):
        if 3 < len(tokens) < 8:
            path = tokens[1].strip('"')
            n0 = 0
            n1 = 0
            dgt = 1
            if tokens[2] == 'exact':
                n0 = int(tokens[3])
                n1 = n0 + 1
                dgt = 3
            elif tokens[2] == 'range':
                n0 = int(tokens[3])
                n1 = int(tokens[4])
                dgt = 4
            else:
                return self.help_analyze()
            backup_bytes = 0
            if len(tokens) > dgt + 1:
                if tokens[dgt + 1] == 'backup':
                    if len(tokens) > dgt + 2:
                        backup_bytes = int(
                            tokens[dgt + 2])
                    else:
                        backup_bytes = 100000
                else:
                    return self.help_analyze()
            print(f'{path}, {n0}, {n1}, {backup_bytes}')
            self.analysis = Markov_Analysis(
                path, n0, n1, backup_bytes=backup_bytes)
        else:
            return self.help_analyze()

    def help_analyze(self):
        print('analyze <path> exact <n> [backup [<bytes]]')
        print('analyze <path> range <n0> <n1> [backup [<bytes]]')

    def cmd_resume(self, tokens):
        if len(tokens) == 2:
            self.analysis = Markov_Analysis(
                path=tokens[1], resume=True)
        else:
            return self.help_resume()

    def help_resume(self):
        print('resume <path>')

    def cmd_read(self, tokens):
        if len(tokens) == 2:
            path = tokens[1].strip('"')
            self.active_chain = Markov_Chain(path)
            # print(self.active_chain.tree)
        else:
            return self.help_read()

    def help_read(self):
        print('read <path>')

    def cmd_random(self, tokens):
        if self.active_chain == None:
            self.msg_no_chain()
        if 3 < len(tokens) < 8:
            count = int(tokens[1])
            l0 = 0
            l1 = 10000
            dgt = 0
            if tokens[2] == 'min':
                l0 = int(tokens[3])
                dgt = 3
            elif tokens[2] == 'max':
                l1 = int(tokens[3])
                dgt = 3
            elif tokens[2] == 'exact':
                l0 = int(tokens[3])
                l1 = l0 + 1
                dgt = 3
            elif tokens[2] == 'range':
                l0 = int(tokens[3])
                l1 = int(tokens[4])
                dgt = 4
            else:
                return self.help_random()
            word = Markov_Word(self.active_chain, l0, l1)
            file = None
            if len(tokens) > dgt + 1:
                if tokens[dgt + 1] == 'save':
                    file = open(tokens[dgt + 2].strip('"'), 'w+')
                else:
                    return self.help_random()
            print(f'[STA] Calculating {count} Random Words')
            for _ in range(count):
                w = word.random()
                print('[OUT] {:<40}'.format(w))
                if file != None:
                    file.write(f'{w}\n')
            if file != None:
                file.close()
        else:
            return self.help_random()

    def help_random(self):
        print('random <count> min <l0> [save <path>]')
        print('random <count> max <l1> [save <path>]')
        print('random <count> exact <l> [save <path>]')
        print('random <count> range <l0> <l1> [save <path>]')

    def cmd_build(self, tokens):
        if not 2 < len(tokens) < 4:
            return self.help_build()
        path = tokens[1].strip('"')
        n = int(tokens[2])
        self.active_chain = Markov_Builder(self, path, n).run()
        print('[STA] Your Active Chain was set')

    def help_build(self):
        print('build <path> <n>')

    def msg_no_chain(self):
        print('Please analyze or read a file first.')
        self.help_analyze()
        self.help_read()

    def remove_line(self):
        print('                                                                            ', end='\r')
