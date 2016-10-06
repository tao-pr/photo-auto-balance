"""
Convolutional Neural Network
---
@author TaoPR (github.com/starcolon)
"""

from theano import *
from theano import tensor as T
from scipy import *
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.objectives import *
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from termcolor import colored
from . import *

class CNN():

  """
  @param {int} dimension of feature vector
  """
  def __init__(self, image_dim, final_vec_dim):

    l1 = ('input',   layers.InputLayer)
    l2 = ('conv1',   layers.Conv2DLayer)
    l3 = ('pool1',   layers.MaxPool2DLayer)
    l4 = ('conv2',   layers.Conv2DLayer)
    l5 = ('pool2',   layers.MaxPool2DLayer)
    l6 = ('hidden1', layers.DenseLayer)
    l7 = ('output',  layers.DenseLayer)

    # Create a NN structure
    print('...Building initial structure')
    print('...input size of  : ', image_dim)
    print('...output size of : ', final_vec_dim)
    self.net = NeuralNet(
      layers=[l1, l2, l3, l4, l5, l6, l7],
      input_shape=(None,) + image_dim,
      conv1_num_filters=50, conv1_filter_size=(5, 5), 
      pool1_pool_size=(3, 3),
      conv2_num_filters=50, conv2_filter_size=(5, 5),
      pool2_pool_size=(3, 3),
      hidden1_num_units=256,
      hidden2_num_units=64,
      output_num_units=final_vec_dim, output_nonlinearity=None,
      update_learning_rate=0.01,
      update_momentum=0.8,
      regression=True,
      max_epochs=200,
      objective_loss_function=squared_error,
      verbose=1
      )

  def train(self,X,y):
    self.net.fit(X,y)

  def predict(self,candidates):
    return self.net.predict(candidates)



