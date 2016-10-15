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
from lasagne.updates import adagrad
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
    input_var  = T.tensor4('X')
    target_var = T.dvector('y')
    input_dim  = (None,) + image_dim

    l_input = layers.InputLayer(shape=input_dim)
    l_conv1 = layers.Conv2DLayer(l_input, 32, (5,5))
    l_conv2 = layers.Conv2DLayer(l_conv1, 32, (3,3))
    l_pool  = layers.MaxPool2DLayer(l_conv2, (5,5), stride=2)
    l_1d1   = layers.DenseLayer(l_pool, 64)
    l_1d2   = layers.DenseLayer(l_hid1, 48)

    self.net       = l_1d2
    self.input_var = l_input.input_var


  # DEPRECATED:
  def old_init(self):
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
      conv1_num_filters=16, conv1_filter_size=(7, 7), 
      pool1_pool_size=(3, 3),
      conv2_num_filters=10, conv2_filter_size=(3, 3),
      pool2_pool_size=(3, 3),
      hidden1_num_units=128,
      output_num_units=final_vec_dim, output_nonlinearity=None,
      update_learning_rate=0.01,
      # update_momentum=0.8,
      regression=True,
      max_epochs=150,
      objective_loss_function=squared_error,
      verbose=1
      )


  # def train(self,X,y):
  #   self.net.fit(X,y)

  def train(self,X,y,num_epochs=100,learn_rate=0.01):
    
    output = get_output(self.net)

    # RMSE loss
    print(colored('...Preparing measurement functions','green'))
    loss   = T.mean((output - y)**2)
    params = get_all_params(self.net)
    update = adagrad(loss, params, learn_rate)

    print(colored('...Preparing training functions','green'))
    train  = theano.function(
      [self.input_var, y],
      loss, updates=update
      )

    try:
      for epoch in range(num_epochs):
        print('...Ep #', epoch)
        train(X, y)
        cost = loss()
    except:
      # Any error
      print(colored('ERROR:','red'))
      pass


    # >>> updates_sgd = sgd(loss, params, learning_rate=0.0001)
    # >>> updates = apply_momentum(updates_sgd, params, momentum=0.9)
    # >>> train_function = theano.function([x, y], updates=updates)


    # train = theano.function([l_in.input_var, target_values, l_mask.input_var],
    #                         cost, updates=updates)
    # compute_cost = theano.function(
    #     [l_in.input_var, target_values, l_mask.input_var], cost)


    # print("Training ...")
    # try:
    #     for epoch in range(num_epochs):
    #         for _ in range(EPOCH_SIZE):
    #             X, y, m = gen_data()
    #             train(X, y, m)
    #         cost_val = compute_cost(X_val, y_val, mask_val)
    #         print("Epoch {} validation cost = {}".format(epoch, cost_val))
    # except KeyboardInterrupt:
    #     pass


  def predict(self,candidates):
    return self.net.predict(candidates)



