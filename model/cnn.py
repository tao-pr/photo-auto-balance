"""
Convolutional Neural Network
---
@author TaoPR (github.com/starcolon)
"""

from theano import *
from theano import tensor as T
from scipy import *
import numpy as np
from . import *

# Create an empty structure of a CNN model
def init_models(num_target_vals):
  X = T.fmatrix('X')
  Y = T.fvector('Y')
  pass


