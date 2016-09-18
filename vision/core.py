"""
Core module for computer visionary stuffs
---
@author TaoPR (github.com/starcolon)
"""

import numpy as np
from PIL import *

def generate_filtered(sample):
  raise NotImplementedError

def load_img(path):
  return array(Image.open(path)).convert('L')