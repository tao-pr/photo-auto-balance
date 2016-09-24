"""
Fundamental of neural network
---
@author TaoPR (github.com/starcolon)
"""

from theano import *
from theano import tensor as T
from scipy import *
import numpy as np

class HiddenNodes():
  """
  @param {Int} dimensionality of input matrix
  @param {Int} dimensionality of output matrix (number of next hidden units)
  @param {T.Opr} activation function
  @param {np.array 2dim} initial node weight: (#dimX, #dimY)
  @param {np.array} initial b vector
  """
  def __init__(self, X_dim, Y_dim, activation=T.tanh, W=None, b=None):
    self.activation = activation
    self.W = W if W is not None else __init_weight(seed, X_dim, Y_dim)
    self.b = b if b is not None else __init_b(Y_dim)


  def __init_weight(self, X_dim, Y_dim):
    # Initial weight based on [Xavier10]'s study
    W_vec = np.asarray(np.random.uniform(\
      -np.sqrt(6.0/(X_dim+Y_dim)),\
      np.sqrt(6.0/(X_dim+Y_dim)),\
      (X_dim, Y_dim)),\
    dtype=config.floatX)

    # For sigmoid activation, use 4 times higher weight
    if self.activation == T.nnet.sigmoid:
      W_vec *= 4.0

    return shared(value=W_vec, name='W', borrow=True)


  def __init_b(self, Y_dim):
    b_vec = np.zeros((Y_dim,), dtype=config.floatX)
    return shared(value=b_vec, name='b', borrow=True)


  """
  Produce an output
  @param {T.dmatrix} input matrix :(#sample, #dimX)
  """
  def generate(self,X):
    out = T.dot(X, self.W) + self.b
    if self.activation is None: out
    else: self.activation(out)




