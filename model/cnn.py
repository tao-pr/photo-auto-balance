"""
Convolutional Neural Network
---
@author TaoPR (github.com/starcolon)
"""

from theano import *
from theano import tensor as T
from scipy import *
import theanets as N
import theanets.layers.convolution as C
import numpy as np
from . import *

class CNN():

  """
  @param {(Int,Int)} dimension of images
  """
  def __init__(self, image_dim, final_vec_dim)

    w,h = image_dim

    l1 = C.Conv1(w*h)         # Photo scanner
    l2 = C.Conv1(round(l1/2)) # Shape downsampling
    l3 = C.Conv1(l2-2)        # Encoder
    l4 = (round(l3/2), 'sigmoid') # Final feature mapper
    l5 = (final_vec_dim, 'linear') # Classifiers

    # Create a NN structure
    self.net = N.regressor(\
      layers=[l1, l2, l3, l4, l5]\
      )

    # Adjust parameters
    self.add_loss('xe', weight=0.1) # Cross-entropy loss


  def train(self,trainset,validationset):
    # self.net.train(
    #   trainset, validationset,
    #   algo='rmsprop',
    #   learning_rate=0.01 )
    n = 0
    for train,valid in self.net.itertrain(trainset, validationset, **kwargs):
      print('...Training iter #{0}, loss = {1:.2f}'.format(n, train['loss']))
      n += 1

  def predict(self,candidate):
    return self.net.predict(candidate)



