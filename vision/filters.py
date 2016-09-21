"""
Image filters
---
@author TaoPR (github.com/starcolon)
"""

import numpy as np

FILTERS = [] # TAOTODO: List all predefined filters here

# Apply multiple filters at once 
# by related supplying arguments in
def filter(brightness_scale,level_a,level_c,hue_deg,sat_scale):
  def f(img):
    hsv  = img if img.mode=='HSV' else img.convert('HSV')
    hsv_ = hsv if brightness_scale == 1 else brightness(brightness_scale)(hsv)
    hsv_ = hsv_ if abs(level_a | level_c) == 0 else level(level_a, level_c)(hsv_)
    hsv_ = hsv_ if hue_deg == 0 else hue(hue_deg)(hsv_)
    hsv_ = hsv_ if sat_scale == 1 else sat(sat_scale)(hsv_)
    return hsv_

  return f

# Generate random filters
def random_filters(max_product_count):
  def rand_f():
    brightness_scale = np.random.normal(1,0.3)
    level_a   = np.random.normal(0,0.25)
    level_c   = np.random.normal(0,0.3)
    hue_deg   = np.random.normal(0,30)
    sat_scale = np.random.normal(1,0.25)
    return filter(brightness_scale,level_a,level_c,hue_deg,sat_scale)

  f = [rand_f() for i in range(max_product_count)]
  return f

# Adjust intensity of an image, such that
# I' = scale * I
def brightness(scale):
  def f(img):
    if img.mode != 'HSV':
      img = img.convert('HSV')
    h,s,v = img.split()
    v     = v.point(lambda x: x*scale)
    return Image.merge('HSV',[h,s,v])
  return f

# Adjust image level, such that
# I' = a*I + c
def level(a,b,c):
  def f(img):
    return img.point(lambda x: x*a+c)
  return f

# Adjust the hue of an image, such that
# h' = h + deg
def hue(deg):
  def f(img):
    if img.mode != 'HSV':
      img = img.convert('HSV')
    h,s,v = img.split()
    h     = h.point(lambda x: x+deg)
    return Image.merge('HSV',[h,s,v])
  return f

# Adjust the saturation of an image, such that
# s' = s * scale
def sat(scale):
  def f(img):
    if img.mode != 'HSV':
      img = img.convert('HSV')
    h,s,v = img.split()
    s     = s.point(lambda x: x*scale)
    return Image.merge('HSV',[h,s,v])
  return f