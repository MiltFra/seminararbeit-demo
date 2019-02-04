#!/usr/bin/env python3
#! Version 3

from markov_interface import Markov_Interface
import sys

name = '[MARKOV 2018-09-10]'
interface = Markov_Interface()

if len(sys.argv) == 1:
    print(f'{name} ', end='')
    while interface.process(input()):
        print(f'{name} ', end='')
elif len(sys.argv) == 2:
    with open(sys.argv[1]) as f:
        l = f.readline().strip('\n')
        while l != "":
            interface.process(l)
            l = f.readline().strip('\n')
else:
    interface.process(" ".join(sys.argv[1:]))
