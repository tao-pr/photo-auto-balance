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
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from termcolor import colored
from . import *

class CNN():

  """
  @param {int} dimension of feature vector
  """
  def __init__(self, image_dim, final_vec_dim):

    n1 = round(image_dim/16)
    n2 = round(n1/8)
    n3 = final_vec_dim

    print('...CNN layers nodes:')
    print('...input dim: {0}'.format(image_dim))
    print('...layer #1 : {0} nodes'.format(n1))
    print('...layer #2 : {0} nodes'.format(n2))
    print('...layer #3 : {0} nodes'.format(n3))

    l1 = ('input',   layers.InputLayer)
    l2 = ('conv2d1', layers.Conv2DLayer)
    l3 = ('pool1',   layers.MaxPool2DLayer)
    l4 = ('hidden1', layers.DenseLayer)
    l5 = ('output',  layers.DenseLayer)



    # Create a NN structure
    print('...Building initial structure')
    self.net = NeuralNet(
      layers=[l1, l2, l3, l4, l5],
      input_shape=(None, 1, image_dim, image_dim*3),
      conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
      hidden4_num_units=400,
      output_num_units=final_vec_dim,
      update_learning_rate=0.1,
      update_momentum=0.8,
      regression=True,
      max_epochs=200,
      verbose=1
      )

  def train(self,X,y):
    print(colored('Training CNN','green'))
    self.net.fit(X,y)
    print(colored('Training all DONE!','green'))

  def predict(self,candidate):
    return self.net.predict(candidate)



