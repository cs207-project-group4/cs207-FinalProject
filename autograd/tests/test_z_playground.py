# -*- coding: utf-8 -*-

from autograd.blocks.trigo import sin, cos
from autograd.variable import Variable
import numpy as np
import autograd as ad
from autograd import config





class foo():
    def __init__(self, x):
        self.x=x
        
    def __sub__(self, other):
        if type(other)==foo:
            return(self.x-other.x)
        else:
            return(self.x-other)
            
    def __rsub__(self, other):
        return(other-self.x)
        
m=foo(10)
p=foo(5)

print(5-m)


#
#
#ad.set_mode('reverse')
#x=Variable([1,2,3])
#
#
#block=sin()
#block2=cos()
#
##for i in range(100):
#    
#t=block(x)
#z=block2(x)
#
#u=t*z
#
#u.commpute_gradients()
#print(u.gradient)
#
#ad.reset_graph()
#
#u.commpute_gradients()
#
#print(u.gradient)
#class test():
#    def __init__(self,x):
#        print('initi 1')
#        self.x=x
#        
#    @classmethod
#    def make_several(self,x,y):
#        t1=self(x)
#        t2=self(y)
#        return(t1,t2)
#        
#





