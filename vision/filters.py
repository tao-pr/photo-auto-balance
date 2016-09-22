"""
Image filters
---
@author TaoPR (github.com/starcolon)
"""

from PIL import Image
import numpy as np

# Apply multiple filters at once 
# by related supplying arguments in
# @param {list} parameter vector
# @param {bool} apply as inverse of transformation?
def transform(v,inversed=False):
  brightness_scale,r,hue_deg,sat_scale = v \
    if not inversed \
    else inverse(v)
      
  def f(img):
    hsv  = img if img.mode=='HSV' else img.convert('HSV')
    hsv_ = hsv if brightness_scale == 1 else brightness(brightness_scale)(hsv)
    hsv_ = hsv_ if r == 1 else gamma(r)(hsv_)
    hsv_ = hsv_ if hue_deg == 0 else hue(hue_deg)(hsv_)
    hsv_ = hsv_ if sat_scale == 1 else sat(sat_scale)(hsv_)
    return hsv_
  return f

def batch_transform(vs,sample):
  return [transform(v)(sample) for v in vs]

# Make an inverse of a transformation vector
def inverse(v):
  brightness_scale,r,hue_deg,sat_scale = v
  return [\
    1/brightness_scale,\
    1/r,\
    -hue_deg,\
    1/sat_scale]


# Generate random filters
# as a list of transformation vectors
def random_filters(max_product_count):
  def rand_f():
    brightness_scale = np.random.normal(1,0.3)
    r                = np.random.normal(1,0.25)
    hue_deg          = np.random.normal(0,30)
    sat_scale        = np.random.normal(1,0.25)
    return [brightness_scale,r,hue_deg,sat_scale]

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

# Gamma transformation of an image such that
# I' = MAX * (I/255)^r
def gamma(r):
  g = lambda x: 255.0*(x/255.0)**r
  def f(img):
    if img.mode != 'RGB':
      return img.point(g)
    else: # Presumably only HSV
      h,s,v = img.split()
      v     = v.point(g)
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