"""
Image filters
---
@author TaoPR (github.com/starcolon)
"""

FILTERS = [] # TAOTODO: List all predefined filters here

# Adjust intensity of an image, such that
# I' = scale * I
def brightness(scale):
  def f(img):
    return img.point(lambda x: x*scale)
  return f

# Adjust image level, such that
# I' = a*I + c
def level(a,c):
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