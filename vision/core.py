"""
Core module for computer visionary stuffs
---
@author TaoPR (github.com/starcolon)
"""

import numpy as np
from PIL import Image
from .filters import *

def get_sample_dim(): # n
  return 240 # total vector dimensionality : n^2 * 3

def generate_filtered(sample,max_combination=5):
  # Make transformation functions
  ts = random_filters(max_combination)
  # Apply generated transformations
  results = [transform(t)(sample) for t in ts]
  return ts, results

def load_img(path):
  return Image.open(path)

# Make a 2D feature vector
def img_to_feature(img):
  d   = min(img.size)
  img = img.crop((0,0,d,d)) # Make it square
  img = img.resize((get_sample_dim(),get_sample_dim())) # Unify the dimension
  img = img if img.mode=='HSV' else img.convert('HSV')

  # Normalise each channel
  h,s,v = img.split()
  h = np.multiply(h, 1.0/360)
  s = np.multiply(s, 1.0/255)
  v = np.multiply(v, 1.0/255)

  return np.array([h,s,v])

def inverse_trans(v):
  return inverse(v)

# Apply a sample image with a transformation filter
def apply_filter(v,inverse=False):
  def to(sample):
    return transform(v,inverse)(sample)
  return to