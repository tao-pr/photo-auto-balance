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
import numpy as np
from itertools import tee
from termcolor import colored
from pprint import pprint
from vision.core import *
from model.core import *

arguments = argparse.ArgumentParser()
arguments.add_argument('--debug', dest='debug', action='store_true')
arguments.add_argument('--limit', type=int, default=None) # Limit the number of samples to process
arguments.add_argument('--dir', type=str, default='./data/raw/') # Where to pick the samples
arguments.add_argument('--train', dest='train', action='store_true') # Training mode
arguments.add_argument('--ratio', type=float, default=0.8) # Ratio of the training set for cross validation
arguments.add_argument('--permutation', type=int, default=8) # Number of filters to apply
arguments.add_argument('--enhance', dest='enhance', action='store_true') # Validation mode
args = vars(arguments.parse_args(sys.argv[1:]))

def train(samples):
  fullset = []
  sample_text_file = '{0}/../trainset.csv'.format(args['dir'])
  
  # Generate trainset samples from the given input dir
  n = 0
  with open(sample_text_file,'w+') as sf:
    for s in samples:
      n += 1
      print(colored('Processing ... ','cyan'), s)
      src     = load_img(args['dir'] + '/' + s)
      dir_out = args['dir'] + '../out'
      
      # Apply generated filters
      transformations,filtered = generate_filtered(src,args['permutation'])
      print('...{} filters applied'.format(\
        colored(len(filtered),'green')))

      # Save generated filtered images
      print(colored('...Saving outputs','yellow'))
      for i in range(len(filtered)):
        name = s.split('.')[0]
        # Save filtered images
        filtered[i]\
          .convert('RGB')\
          .save('{0}/{1}-{2:02d}.jpg'.format(dir_out,name,i))

        # Save the trainset with following fields
        # filename | # | inverse transformation vector
        inv_trans_vec = ','.join([str(k) for k in inverse_trans(transformations[i])])
        sf.write("{0},{1},{2}\n".format(s,i,inv_trans_vec))
        fullset.append({'x':img_to_feature(filtered[i]), 'y':transformations[i]})
  

      # TAODEBUG: Save invert-filtered images
      if args['debug']: 
        dir_out2 = args['dir'] + '../unfiltered'
        for i in range(len(filtered)):
          name = s.split('.')[0]
          unfiltered = apply_filter(transformations[i],inverse=True)(filtered[i])
          unfiltered\
            .convert('RGB')\
            .save('{0}/{1}-{2:02d}.jpg'.format(dir_out2,name,i))
    

      if args['limit'] and n>=args['limit']:
        print(colored('LIMIT REACHED','yellow'))
        break

  # Process the prepared trainset
  if len(fullset)>0:
    print(colored('Training started ...','magenta'))
    np.random.shuffle(fullset)
    # Split into actual trainset and validation set
    print('...{0} raw images processed'.format(n))
    print('...{0} samples to go'.format(len(fullset)))
    d = round(args['ratio']*len(fullset))
    print('...{0} for training'.format(d))
    print('...{0} for validation'.format(len(fullset)-d))

    print('...Reading samples')
    trainsetX = np.asarray([l['x'] for l in fullset[d:]])
    validsetX = np.asarray([l['x'] for l in fullset[:d]])
    trainsetY = np.asarray([l['y'] for l in fullset[d:]])
    validsetY = np.asarray([l['y'] for l in fullset[:d]])

    shape_x = np.shape(fullset[0]['x'])
    shape_y = np.shape(fullset[0]['y'])
    cnn = train_model(
      trainsetX,
      trainsetY,
      validsetX,
      validsetY,
      shape_x, 
      shape_y[0])

    # Serialise the model
    path_model = args['dir'] + '../model.cnn'
    save_model(cnn, path_model)

  else:
    print(colored('No samples in the given directory.','yellow'))
    print(colored('Ending the job now...','yellow'))

def enhance(samples):
  # Load the samples, convert them to feature vectors
  print(colored('Loading samples...','green'))
  S = [load_img(args['dir'] + '/' + s) for s in samples]
  X = [img_to_feature(s) for s in S]

  print(np.shape(X))

  # Load the model
  path_model = args['dir'] + '/../model.cnn'
  model = load_model(path_model)

  # Generate transformation vectors for those samples
  V = model.predict(X)

  print(np.shape(V))

  # Generate outputs 
  print(colored('Generating outputs...','magenta'))
  for s,u,v in zip(samples,S,V):
    path_out = args['dir'] + '/../unfiltered/' + s
    print('...Processing : {0}'.format(colored(s,'cyan')))
    out = apply_filter(v)(u)
    out.convert('RGB').save(path_out)

  print('...Done!')

def process_with(path, f):
  # Iterate through the path
  dataset = [f for f in os.listdir(path) if '.jpg' in f]
  return f(dataset)

if __name__ == '__main__':
  print(colored('=====================','magenta'))
  print(colored('Loader started ...','magenta'))
  print(colored('=====================','magenta'))
  
  path      = args['dir']
  print('Executing function...')
  with_func = train if args['train'] else enhance
  process_with(path, with_func)
