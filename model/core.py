"""
Model training and utilisation 
---
@author TaoPR (github.com/starcolon)
"""

from .cnn import CNN
from termcolor import colored
import _pickle as pickle
import numpy as np
from lasagne.objectives import *

"""
Train the CNN model with the given dataset
@param {String/None} path to the existing saved model (if any)
@param {X-trainset}
@param {y-trainset}
@param {X-validation}
@param {y-validation}
@param {(int,int)} dimentions of image
@param {int} dimension of output vector
@param {int} number of epochs
@param {int} batch size
"""
def train_model(path, X, y, X_, y_, image_dim, final_vec_dim, epoch, batch_size):

  # Create a new CNN
  if path is None or not os.path.isfile(path + '0'):
    print(colored('Creating a new CNN.','cyan'))
    cnn = CNN(image_dim, final_vec_dim) 
  else:
    print(colored('Loading model from : {0}'.format(path),'green'))
    cnn = load_model(path, image_dim, final_vec_dim)
  
  # Train the network
  print(colored('Training started.','green'))
  cnn.train(X, y, X_, y_, batch_size, epoch)
  print(colored('Training finished.','green'))

  return cnn

def save_model(cnn,path):
  cnn.save(path)

def load_model(path, image_dim, final_vec_dim):
  return CNN.load(path, image_dim, final_vec_dim)

def generate_output(cnn,candidate):
  return cnn.predict(candidate)