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
from vision.core import *
from model.core import *

arguments = argparse.ArgumentParser()
arguments.add_argument('--dir', type=str, default='./data/') # Where to pick the samples
arguments.add_argument('--train', dest='train', action='store_true') # Training mode
arguments.add_argument('--validate', dest='validate', action='store_true') # Validation mode
args = vars(arguments.parse_args(sys.argv[1:]))

def train(samples):
  trainset = []
  n = 0 # Number of input samples
  s = 0 # Number of generated cases
  for s in samples:
    print(colored('Processing ... ','cyan'), s)
    src = load_img(s)
    
    # Apply generated filters
    filtered = generate_filtered(src)
    print('    ${} filters applied'.format(len(filtered)))

    # Aggregate training set
    # TAOTODO: Make this a streamable generator?
    n += 1
    s += len(filtered)
    trainset.push((src, filtered))

  # Pass the trainset through the training process
  if n>0:
    print(colored('Training started ...','magenta'))
    lm = train_model(trainset)
    # TAOTODO: Report the training results

    
  else:
    print(colored('No samples in the given directory.','yellow'))
    print(colored('Ending the job now...','yellow'))

def validate(samples):
  raise NotImplementedError

def process_with(path, f):
  # Iterate through the path
  dataset = os.listdir(path)
  return f(dataset)

if __name__ == '__main__':
  print(colored('•••••••••••••••••••••','magenta'))
  print(colored('Loader started ...','magenta'))
  print(colored('•••••••••••••••••••••','magenta'))
  
  path      = args['dir']
  print('Executing function...')
  with_func = train if args['train'] else validate
  process_with(path, with_func)
