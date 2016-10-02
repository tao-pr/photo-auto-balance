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
  print(colored('Training finished.','green'))

  # Apply cross validation on (X_,y_)
  print(colored('Cross validation started.','green'))
  z  = cnn.predict(X)
  z_ = cnn.predict(X_)

  # Measure RMS error
  rmse  = np.sum([(a-b)**2 for a,b in zip(z,y)]) / float(len(z))
  rmse_ = np.sum([(a-b)**2 for a,b in zip(z_,y_)]) / float(len(z_))

  print('===============================================')
  print(' RMS Error measured on trainset:       {0:.2f}'.format(rmse))
  print(' RMS Error measured on validation set: {0:.2f}'.format(rmse_))
  print('===============================================')

  return cnn

def save_model(cnn,path):
  with open(path, 'wb') as f:
    print(colored('Saving the model','green'))
    pickle.dump(cnn, f, -1)
    print('...Done!')

def load_model(path):
  with open(path, 'rb') as f:
    print(colored('Loading the model','green'))
    return pickle.load(f)

def generate_output(cnn,candidate):
  return cnn.predict(candidate)