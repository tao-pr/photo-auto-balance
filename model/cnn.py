"""
Convolutional Neural Network
---
@author TaoPR (github.com/starcolon)
"""

from theano import *
from theano import tensor as T
from scipy import *
import theanets as N
import theanets.layers.convolution as Cvl
import numpy as np
from . import *

class CNN():

  """
  @param {int} dimension of feature vector
  """
  def __init__(self, image_dim, final_vec_dim):

    n1 = round(image_dim/3)
    n2 = round(n1/8)
    n3 = round(n2/16)
    n4 = final_vec_dim

    print('...CNN layers nodes:')
    print('...input dim: {0}'.format(image_dim))
    print('...layer #1 : {0} nodes'.format(n1))
    print('...layer #2 : {0} nodes'.format(n2))
    print('...layer #3 : {0} nodes'.format(n3))
    print('...layer #4 : {0} nodes'.format(n4))

    l1 = Cvl.Conv1(n1,size=n1,inputs=image_dim)  # Pixel-shape scanner
    l2 = Cvl.Conv1(n2,size=n2,inputs=n1)  # Shape encoder
    l3 = Cvl.Conv1(n3,size=n3,inputs=n2)  # Final feature mapper
    l4 = Cvl.Conv1(n4,size=n4,inputs=n3)  # Classifiers

    # Create a NN structure
    self.net = N.Autoencoder(\
      layers=[l1, l2, l3, l4]\
      )

    # Adjust parameters
    self.add_loss('xe', weight=0.1) # Cross-entropy loss


  def train(self,trainset,validationset):
    print(colored('Training CNN','green'))
    self.net.train(
      trainset, validationset,
      algo='rmsprop',
      learning_rate=0.01 )
    n = 0
    # for train,valid in self.net.itertrain(trainset, validationset, **kwargs):
    #   print('...Training iter #{0}, loss = {1:.2f}'.format(n, train['loss']))
    #   n += 1

  def predict(self,candidate):
    return self.net.predict(candidate)



