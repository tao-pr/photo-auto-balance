"""
Model training and utilisation 
---
@author TaoPR (github.com/starcolon)
"""

from .cnn import CNN
from termcolor import colored
import _pickle as pickle
import numpy as np

"""
Train the CNN model with the given dataset
@param {X-trainset}
@param {y-trainset}
@param {X-validation}
@param {y-validation}
@param {(int,int)} dimentions of image
@param {int} dimension of output vector
"""
def train_model(X, y, X_, y_, image_dim, final_vec_dim):

  # Create a new CNN
  print(colored('Creating a new CNN.','green'))
  cnn = CNN(image_dim,final_vec_dim)
  
  # Train the network
  print(colored('Training started.','green'))
  cnn.train(X, y)

  # TAOTODO: Apply cross validation on (X_,y_)

  return cnn

def save_model(cnn,path):
  with open(path, 'wb') as f:
    print(colored('Saving the model','green'))
    pickle.dump(cnn, f, -1)
    print('...Done!')

def load_model(path):
  with open(path, 'rb') as f:
    print(colored('Loading the model','green'))
    return pickle.load(f, -1)

def generate_output(cnn,candidate):
  return cnn.predict(candidate)