# -*- coding: utf-8 -*-
import numpy as np

# =============================================================================
# Contains relevant utils fonctions for the AD package
# =============================================================================


def get_shape(x):
    """
    get the shape of the input, set to (1,) if it is a real number
    """
    typeOfInput = type(x)
    
    #if the data provided is already in an 'array' type (numpy or list)
    # such as x=Variable([2,34,4])
    if typeOfInput==np.ndarray or typeOfInput==list:
        #numpy array or python list
        shape=np.array(x).shape
       
         
    else:
        #real input, like x=Variable(2)
        shape=(1,) 
    
    return(shape)
    
    
    
def data_2_numpy(data):
    """
    convert input data (np.array, list, or float) into a numpy array
    """
    typeOfInput = type(data)    
    
    #the data provided is already an array
    if typeOfInput==np.ndarray:
        #numpy array 
        return(data)
        
    #the data provided is a list
    elif typeOfInput == list:
        # python list
        return(np.array(data))
        
    #the data provided is a real number
    else:
        #real input
        return(np.array([data]))
