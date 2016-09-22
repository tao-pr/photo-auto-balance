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
arguments.add_argument('--debug', dest='debug', action='store_true')
arguments.add_argument('--dir', type=str, default='./data/raw/') # Where to pick the samples
arguments.add_argument('--train', dest='train', action='store_true') # Training mode
arguments.add_argument('--permutation', type=int, default=5) # Number of filters to apply
arguments.add_argument('--validate', dest='validate', action='store_true') # Validation mode
args = vars(arguments.parse_args(sys.argv[1:]))

def train(samples):
  trainset = []
  n = 0 # Number of input samples
  for s in samples:
    print(colored('Processing ... ','cyan'), s)
    src = load_img(args['dir'] + '/' + s)
    
    # Apply generated filters
    transformations,filtered = generate_filtered(src,args['permutation'])
    print('    {} filters applied'.format(\
      colored(len(filtered),'green')))

    # TAODEBUG: Save transformed images
    if args['debug']:
      print(colored('     Saving outputs','yellow'))
      dir_out  = args['dir'] + '../out'
      dir_out2 = args['dir'] + '../unfiltered'
      for i in range(len(filtered)):
        name = s.split('.')[0]
        # Save filtered images
        filtered[i]\
          .convert('RGB')\
          .save('{0}/{1}-{2:02d}.jpg'.format(dir_out,name,i))
        # Save invert-filtered images
        unfiltered = apply_filter(transformations[i],inverse=True)(filtered[i])
        unfiltered\
          .convert('RGB')\
          .save('{0}/{1}-{2:02d}.jpg'.format(dir_out2,name,i))

    # Aggregate training set
    # TAOTODO: Make this a streamable generator?
    n += 1
    trainset.append((src, filtered))

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
  dataset = [f for f in os.listdir(path) if '.jpg' in f]
  return f(dataset)

if __name__ == '__main__':
  print(colored('•••••••••••••••••••••','magenta'))
  print(colored('Loader started ...','magenta'))
  print(colored('•••••••••••••••••••••','magenta'))
  
  path      = args['dir']
  print('Executing function...')
  with_func = train if args['train'] else validate
  process_with(path, with_func)
