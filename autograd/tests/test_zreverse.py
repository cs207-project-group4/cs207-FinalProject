# -*- coding: utf-8 -*-

from autograd.blocks.trigo import sin, cos
from autograd.variable import Variable
import numpy as np
import autograd as ad
from autograd import config



#
#ad.mode = 'reverse'
#x=Variable([1,2,3])
#block=sin()
#block2=cos()
#
##for i in range(100):
#    
#y=block(x)
#z=block2(x)
#
#u=z*y
#
#u.backward()
#print(u.gradient)
#
#
#ad.reset_graph()
#t=block(block2(3*y+z))
#u/=t
#u.backward()
#print(u.gradient)
#
#
##ad.reset_graph()








