#!/usr/bin/env python

"""
Load image dataset from physical location 
and apply them to the particular processing pipeline.
---
USAGE:
  $ ./loader.py --dir {dataset} --
---
@author TaoPR (github.com/starcolon)
"""

import os
import sys
import argparse
from itertools import tee
from termcolor import colored
from pprint import pprint

arguments = argparse.ArgumentParser()
arguments.add_argument('--dir', type=str, default='data/') # Where to pick the samples
arguments.add_argument('--train', dest=train, action=store_true) # Training mode
arguments.add_argument('--validate', dest=validate, action=store_true) # Validation mode
args = vars(arguments.parse_args(sys.argv[1:]))

def train(samples):
  pass

def validate(samples):
  pass

def process_with(path, f):
  pass

if __name__ == '__main__':
  path      = args['dir']
  with_func = train if args['train'] else validate
  process_with(path, with_func)
