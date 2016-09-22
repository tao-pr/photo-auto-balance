"""
Core module for computer visionary stuffs
---
@author TaoPR (github.com/starcolon)
"""

import numpy as np
from PIL import Image
from .filters import *
#from . import filters

def generate_filtered(sample,max_combination=5):
  # Make transformation functions
  ts = random_filters(max_combination)
  # Apply generated transformations
  results = [transform(t)(sample) for t in ts]
  return ts, results

def load_img(path):
  return Image.open(path)

# Apply a sample image with a transformation filter
def apply_filter(v,inverse=False):
  def to(sample):
    return transform(v,inverse)(sample)
  return to