# -*- coding: utf-8 -*-
from autograd.blocks.block import Block
from autograd.blocks.block import SimpleBlock

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

    def get_jacobians(self, *args):
        shape=args[0].data.shape[0]
        first_term = np.eye(shape)
        second_term = np.eye(shape)

        return([first_term, second_term])


class subtract(Block):
    """
    subtraction of two vector inputs
    """
    def data_fn(self, *args):
        operator_check(args)
        new_data=np.subtract(args[0].data, args[1].data)
        return(new_data)

    def get_jacobians(self, *args):
        shape=args[0].data.shape[0]
        first_term = np.eye(shape)
        second_term = -np.eye(shape)

        return([first_term, second_term])


class multiply(Block): ### OK
    """
    element-wise multiplication
    """
    def data_fn(self, *args):
        operator_check(args)
        new_data = np.multiply(args[0].data, args[1].data)
        return(new_data)

    def get_jacobians(self, *args):
        first_term = np.diag(args[1].data)
        second_term = np.diag(args[0].data)

        return([first_term, second_term])


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

    def get_jacobians(self, *args):

        assert args[1].data.all() != 0, 'dividing by a zero element in the second input : {}'.format(args[1].data)


        y_inv = np.float_power(args[1].data,-1)

        first_term = np.diag(y_inv)
        second_term = -np.diag(np.multiply(args[0].data, np.power(y_inv,2)))


        return([first_term, second_term])



class power(SimpleBlock):
    """
    element-wise power. second argument is value of power
    (int, float, vector) to apply to first argument
    """
    def data_fn(self, input_var, power_exponent):
        new_data = np.float_power(input_var.data, power_exponent)
        return(new_data)

    def gradient_fn(self, input_var, power_exponent):
        new_grad = power_exponent*input_var.data**(power_exponent-1)

        return(new_grad)


class sum_elts(Block):
    """
    sum the elements of the vector
    """
    def data_fn(self, input_vector):
        new_data = np.sum(input_vector.data)
        return(new_data)


    def get_jacobians(self, *args):
        shape = args[0].data.shape[0]
        jacobian=np.ones((1, shape))

        return([jacobian])


class extract(Block):
    def data_fn(self, input_var, key):

        new_data = input_var.data[key]
        return(new_data)


    def get_jacobians(self, input_var, key):
        shape_of_data = input_var.data.shape[0]

        if type(key)==slice:
            #slice extraction
            number_of_lines_to_take = key.stop - key.start
            jacobian=np.zeros((number_of_lines_to_take, shape_of_data))
            row=0
            for i in range(key.start, key.stop):
                jacobian[row, i]=1
                row+=1

        else:
            #element extraction
            jacobian=np.zeros((1,shape_of_data))
            jacobian[0,key]=1
        return([jacobian])
