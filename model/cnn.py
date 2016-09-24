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

    l1 = C.Conv1(w*h)
    l2 = C.Conv1(round(l1/2))
    l3 = C.Conv1(l2-2)
    l4 = (round(l1/2), 'sigmoid')
    l5 = (final_vec_dim, 'linear')

    # Create a NN structure
    self.net = N.regressor(\
      layers=[l1, l2, l3, l4, l5]\
      )

    # Adjust parameters
    self.add_loss('mae', weight=0.1)


  def train(self,trainset,validationset):
    self.net.train(\
      trainset, validationset,\
      algo='rmsprop',\
      hidden_l1=0.01 )

  def predict(self,candidate):
    return self.net.predict(candidate)



