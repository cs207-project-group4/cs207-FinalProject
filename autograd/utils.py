# -*- coding: utf-8 -*-
import numpy as np


def get_shape(x):
    """
    get the shape of the input, set to 1 if it is a real number
    """
    typeOfInput = type(x)
    
    if typeOfInput==np.ndarray or typeOfInput==list:
        #numpy array or python list
        shape=np.array(x).shape
                
    else:
        #real input
        shape=1 
    
    return(shape)
    
def data_2_numpy(data):
    typeOfInput = type(data)    
    if typeOfInput==np.ndarray:
        #numpy array 
        return(data)
        
    elif typeOfInput == list:
        # python list
        return(np.array(data))
    else:
        #real input
        return(np.array([data]))
