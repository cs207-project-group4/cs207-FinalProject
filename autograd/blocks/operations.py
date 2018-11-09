# -*- coding: utf-8 -*-
from autograd.blocks.block import Block

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
        new_grad = np.add(args[0].gradient, args[1].gradient)
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


class multiply(Block): ### OK
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
    provided that b!=0
    divide the first element by the second, provided that b!=0 : 
        divide(a,b) = a/b
    """
    
    def data_fn(self, *args):
        operator_check(args)
        assert args[1].data.all() != 0, 'dividing by a zero element in the second input : {}'.format(args[1].data)
        
        new_data = np.divide(args[0].data, args[1].data)
        return(new_data)

    def gradient_forward(self, *args):
        """
        quotient rule: (x/y)' = (x'y - xy') / y ** 2
        """
        assert args[1].data.all() != 0, 'dividing by a zero element in the second input : {}'.format(args[1].data)

        operator_check(args)
        first_term = np.multiply(args[0].gradient, args[1].data)
        second_term = np.multiply(args[0].data, args[1].gradient)
        diff= np.subtract(first_term, second_term)
        
        third_term = np.power(args[1].data, 2)
        third_term=1/third_term
        
        return np.multiply(diff, third_term)

class power(Block):
    """
    element-wise power. second argument is value of power
    (int, float, vector) to apply to first argument
    """
    def data_fn(self, input_vector, power_exponent):
        new_data = np.float_power(input_vector.data, power_exponent)
        return(new_data)

    def gradient_forward(self, input_vector, power_exponent):
        """
        power & product rule: (x^n)' = nx'x^(n-1)
        """

        #operator_check(args)
        simple_term=np.float_power(input_vector.data, power_exponent - 1)
        gradient_term=np.multiply(input_vector.gradient, simple_term)
        new_grad = np.multiply(power_exponent , gradient_term)
        return (new_grad)
    

class sum_elts(Block):
    """
    sum the elements of the vector
    """
    def data_fn(self, input_vector):
        new_data = np.sum(input_vector.data)
        return(new_data)

    def gradient_forward(self, input_vector):        

        #operator_check(args)        
        shape=input_vector.data.shape[0]
        jacobian = np.ones((1,shape))
        
        
        
        new_grad = np.dot(jacobian, input_vector.gradient)
        return (new_grad)
    

