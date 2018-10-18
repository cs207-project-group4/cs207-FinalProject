# -*- coding: utf-8 -*-

import numpy as np
import autograd.utils as utils

class Variable():
    def __init__(self,data, gradient=None):
    
             
        self.data=utils.data_2_numpy(data)
        
        data_shape=utils.get_shape(data)
        
        if gradient == None:
            #default value for the gradient
            self.gradient = np.ones(data_shape)
        
        else:
            #SHOULD CHECK THE SHAPES IF THEY MATCH
            self.gradient=utils.data_2_numpy(gradient)     
        
        

    
    def set_data(self, data):
        new_shape= utils.get_shape(data)
        assert new_shape==self.data.shape, 'trying to set data with inconsistent shapes! previous shape : {} -- shape provided {}'.format(self.data.shape, new_shape)
        self.data=data
    
    def set_gradient(self, gradient):
        new_shape= utils.get_shape(gradient)
        assert new_shape==self.gradient.shape, 'trying to set data with inconsistent shapes! previous shape : {} -- shape provided {}'.format(self.gradient.shape, new_shape)
        
        self.gradient=gradient
    
    def __str__(self):
        return('data : {} \ngrad : {}'.format(self.data, self.gradient))
        
    
if __name__=='__main__':
    x=Variable([2,3,4])
    
    print(x.data)
    print(x.gradient)
    
    new_data=np.ones(3)
    x.set_data(new_data)
    print(x.data)
    
    new_grad=43*np.ones(3)
    x.set_gradient(new_grad)
    print(x)

