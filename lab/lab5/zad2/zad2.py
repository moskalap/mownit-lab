
from numpy import float32,around,diag
from numpy.linalg import svd
from matplotlib.pyplot import *
import scipy.misc as misc


image = misc.ascent()

u,s,v = svd(image)
k = 10
#slice
u = u[:,0:k]
v = v[0:k,:]
s = diag(s[0:k])


B = u.dot(s).dot(v)
imsave('test.jpg',B, cmap = 'bone')
