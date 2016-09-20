"""
Core module for computer visionary stuffs
---
@author TaoPR (github.com/starcolon)
"""

import numpy as np
from PIL import *
from . import filters

def generate_filtered(sample,max_combination=5):
  raise NotImplementedError

def load_img(path):
  return array(Image.open(path)).convert('L')

def identical_filter(i):
  raise NotImplementedError

# I' = I * f + c
def apply_filter(sample,f,c):
  raise NotImplementedError