# -*- coding: utf-8 -*-

from block import Block
import numpy as np

## NOT UP TO DATE ##
        

class add(Block):
    def data_fn(self, *args):   
        #ASSERT SHAPE FOR DATA AND GRAD
        new_data = np.add(args[0].data, args[1].data)      
        return(new_data)
    
    
    def gradient_fn(self, *args):
        new_grad = np.add ( args[0].gradient, args[1].gradient ) 
        return(new_grad)
        

class dot(Block):
    
    def data_fn(self, *args):   
        #ASSERT SHAPE FOR DATA AND GRAD
        new_data = np.dot(args[0].data, args[1].data)      
        return(new_data)
    
    
    def gradient_fn(self, *args):
        new_grad = np.add ( np.dot( args[0].gradient, args[1].data ) , np.dot( args[0].data, args[1].gradient ) )
        return(new_grad)
        
class multiply(Block):
    """
    element-wise multiplication
    """
    def data_fn(self, *args):   
        #ASSERT SHAPE FOR DATA AND GRAD
        new_data = np.multiply(args[0].data, args[1].data)      
        return(new_data)
    
    
    def gradient_fn(self, *args):
        first_term = np.multiply(args[0].gradient, args[1].data)
        second_term = np.multiply(args[0].data, args[1].gradient)

        new_grad = np.add ( first_term, second_term ) 
        return(new_grad)
    
        
        
        