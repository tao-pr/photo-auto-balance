"""
Core module for computer visionary stuffs
---
@author TaoPR (github.com/starcolon)
"""

import numpy as np
from PIL import Image
from .filters import *
#from . import filters

def get_sample_dim():
  return 128

def generate_filtered(sample,max_combination=5):
  # Make transformation functions
  ts = random_filters(max_combination)
  # Apply generated transformations
  results = [transform(t)(sample) for t in ts]
  return ts, results

def load_img(path):
  return Image.open(path)

def load_as_feature(path):
  d   = min(img.size)
  img = img.crop((0,0,d,d)) # Make it square
  img = img.resize((get_sample_dim(),get_sample_dim())) # Unify the dimension
  img = img if img.mode=='HSV' else img.convert('HSV')
  v   = np.array(img.getdata) # NOTE: Huge computation
  return np.reshape(v.size) # Make it 1D vector

def inverse_trans(v):
  return inverse(v)

# Apply a sample image with a transformation filter
def apply_filter(v,inverse=False):
  def to(sample):
    return transform(v,inverse)(sample)
  return to