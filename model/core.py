"""
Model training and utilisation 
---
@author TaoPR (github.com/starcolon)
"""

from . import cnn

def train_model(dataset):
  raise NotImplementedError

def save_model(model,path):
  raise NotImplementedError

def load_model(path):
  raise NotImplementedError

def generate_output(model,candidate):
  raise NotImplementedError