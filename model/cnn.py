"""
Convolutional Neural Network
---
@author TaoPR (github.com/starcolon)
"""

import time
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
    input_dim  = (None,) + image_dim

    l_input = layers.InputLayer(shape=input_dim)
    l_conv1 = layers.Conv2DLayer(l_input, 32, (5,5))
    l_conv2 = layers.Conv2DLayer(l_conv1, 32, (3,3))
    l_pool  = layers.MaxPool2DLayer(l_conv2, (5,5), stride=2)
    l_1d1   = layers.DenseLayer(l_pool, 64)
    l_1d2   = layers.DenseLayer(l_1d1, final_vec_dim)

    self.net         = l_1d2
    self.input_layer = l_input


  # DEPRECATED:
  # def old_init(self):
  #   l1 = ('input',   layers.InputLayer)
  #   l2 = ('conv1',   layers.Conv2DLayer)
  #   l3 = ('pool1',   layers.MaxPool2DLayer)
  #   l4 = ('conv2',   layers.Conv2DLayer)
  #   l5 = ('pool2',   layers.MaxPool2DLayer)
  #   l6 = ('hidden1', layers.DenseLayer)
  #   l7 = ('output',  layers.DenseLayer)

  #   # Create a NN structure
  #   print('...Building initial structure')
  #   print('...input size of  : ', image_dim)
  #   print('...output size of : ', final_vec_dim)
  #   self.net = NeuralNet(
  #     layers=[l1, l2, l3, l4, l5, l6, l7],
  #     input_shape=(None,) + image_dim,
  #     conv1_num_filters=16, conv1_filter_size=(7, 7), 
  #     pool1_pool_size=(3, 3),
  #     conv2_num_filters=10, conv2_filter_size=(3, 3),
  #     pool2_pool_size=(3, 3),
  #     hidden1_num_units=128,
  #     output_num_units=final_vec_dim, output_nonlinearity=None,
  #     update_learning_rate=0.01,
  #     # update_momentum=0.8,
  #     regression=True,
  #     max_epochs=150,
  #     objective_loss_function=squared_error,
  #     verbose=1
  #     )


  # def train(self,X,y):
  #   self.net.fit(X,y)

  # Train the neural net
  # @param {Matrix} trainset X
  # @param {Vector} trainset y
  # @param {Matrix} validation set X
  # @param {Vector} validation set y
  # @param {int} batch size
  # @param {int} number of epochs to run
  # @param {double} learning rate (non-negative, non-zero)
  def train(self,X,y,X_,y_,batch_size=100,num_epochs=100,learn_rate=0.03):
    
    # Symbolic I/O of the network
    inputx  = self.input_layer.input_var
    outputy = T.dmatrix('ys')             # Expected output
    output  = layers.get_output(self.net) # Actual output

    print('... X : ', X.shape)
    print('... y : ', y.shape)
    print('... X_ : ', X_.shape)
    print('... y_ : ', y_.shape)

    # Minimising RMSE with Adagradient
    print(colored('...Preparing measurement functions','green'))
    loss   = T.mean((output - outputy)**2)
    params = layers.get_all_params(self.net)
    update = adagrad(loss, params, learn_rate)

    print(colored('...Preparing training functions','green'))
    train  = theano.function(
      [inputx, outputy],
      loss, 
      updates=update
      )
    gen_output = theano.function([inputx], output)

    try:
      print(colored('...Training started','green'))

      for epoch in range(num_epochs):
        print('...[Ep] #', epoch)
        
        t0       = time.time()
        b0,bN,bi = 0, batch_size, 0

        # Train each batch of the input
        while bN < X.shape[0]:
          print('......batch #', bi, ' ({0}~{1})'.format(b0,bN))

          train(X[b0:bN], y[b0:bN])

          # Measure training loss (RMSE)
          _output  = gen_output(X[b0:bN])
          _loss    = np.mean((_output - y[b0:bN])**2)
          # Measure validation loss (RMSE)
          _outputv = gen_output(X_)
          _lossv   = np.mean((_outputv - y_)**2)
          print('......loss on trainset   : {0:0.4f}'.format(_loss))
          print('......loss on validation : {0:0.4f}'.format(_lossv))

          b0 += batch_size
          bN += batch_size
          bi += 1
        
        t1 = time.time()
        print(colored('...{0:.1f} s elapsed, {1} batches processed'.format(t1-t0, bi), 'yellow'))

        # Shuffle the trainset
        print('...Shuffling the trainset')
        rd = np.arange(len(y))
        np.random.shuffle(rd)
        X = X[rd]
        y = y[rd]

    except Exception as e:
      # Any error
      print(colored('ERROR:','red'), type(e))
      print(e)
      # TAOTODO: Save the current model as is
      pass

  def predict(self,candidates):
    gen_output = theano.function([inputx], output)




