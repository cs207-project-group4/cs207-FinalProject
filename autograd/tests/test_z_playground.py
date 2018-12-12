# -*- coding: utf-8 -*-

from autograd.blocks.trigo import sin, cos
from autograd.variable import Variable
import numpy as np
import autograd as ad
from autograd import config



class test():
    def __init__(self, x):
        self.x=x
    
    def parent(self):
        print('parent')
        
        
        
class sub(test):
    def __init__(self, x):
        super().__init__(x)
        
    def parent(self):
        print('child')
        super().parent()
        
        
t=sub(2)

t.parent()
