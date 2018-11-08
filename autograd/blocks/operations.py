# -*- coding: utf-8 -*-

from block import Block
import numpy as np
# ======================
#      Operators 
# ======================

def operator_check(*args):
    """
    assertions for inputs passed to operator-type blocks
    """
#    assert len(args)==2, "This block takes only  two inputs, {} were given".format(len(args))
    pass


class add(Block):
    """
    addition of two vector inputs
    """
    def data_fn(self, *args):
        operator_check(args)
        new_data = np.add(args[0].data, args[1].data)      
        return(new_data)
    
    def gradient_forward(self, *args):
        """
        (x + y)' = x' + y'
        """
        operator_check(args)
        new_grad = np.add ( args[0].gradient, args[1].gradient ) 
        return(new_grad)
        

class subtract(Block):
    """ 
    subtraction of two vector inputs
    """
    def data_fn(self, *args):
        operator_check(args)
        new_data=np.subtract(args[0].data, args[1].data)
        return(new_data)
    
    def gradient_forward(self, *args):
        """
        (x - y)' = x' - y'
        """
        operator_check(args)
        new_grad = np.subtract(args[0].gradient, args[1].gradient)
        return(new_grad)
        
        
class multiply(Block): ### SOMETHING WRONG? CHECK DOT
    """
    element-wise multiplication
    """
    def data_fn(self, *args):   
        operator_check(args)
        new_data = np.multiply(args[0].data, args[1].data)      
        return(new_data)
    
    
    def gradient_forward(self, *args):
        """
        product rule: (xy)' = x'y + y'x
        """
        operator_check(args)
        first_term = np.multiply(args[0].gradient, args[1].data)
        second_term = np.multiply(args[0].data, args[1].gradient)
        new_grad = np.add(first_term, second_term) 
        return(new_grad)
    
class divide(Block):
    """
    element-wise division
    """
    def data_fn(self, *args):
        operator_check(args)
        new_data = np.divide(args[0].data, args[1].data)      
        return(new_data)
    
    def gradient_forward(self, *args):
        """
        quotient rule: (x/y)' = (x'y - xy') / y ** 2
        """
        operator_check(args)
        first_term = np.multiply(args[0].gradient, args[1].data) 
        second_term = np.multiply(args[0].data, args[1].gradient)
        third_term = np.power(args[1].data, 2)
        return (first_term - second_term) / third_term   

class power(Block):
    """
    element-wise power. second argument is value of power 
    (int, float, vector) to apply to first argument
    """
    def data_fn(self, *args):
        operator_check(args)
        new_data = np.power(args[0].data, args[1].data)
        return(new_data)
    
    def gradient_forward(self, *args):
        """
        power & product rule: (x^n)' = nx'x^(n-1) 
        """
        operator_check(args)
        new_grad = args[1].data * args[0].gradient * np.power(args[0].data, args[1].data - 1)
        return (new_grad)
        
    
class dot(Block):
    """
    dot product between two vector inputs
    """
    def data_fn(self, *args):
        operator_check(args)
        new_data = np.dot(args[0].data, args[1].data)      
        return(new_data)
    
    def gradient_forward(self, *args):
        """
        dot product returns a scalar
        """
        raise NotImplementedError
        