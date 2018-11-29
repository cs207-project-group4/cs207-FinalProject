# -*- coding: utf-8 -*-

import numpy as np
import autograd.utils as utils
class Variable():
    """
    The variable class is the main class that will cary the information flow : data and gradient
    A variable has a data attribute, and a gradient one
    The data attribute is the data contained is this variable, for instance:
        x=Variable(1)
        y=ad.sin(x)
        
        y is a variable and y.data=sin(1)
        
    The gradient attribute contains the Jacobian matrix that gathers all the gradient flow
    from the beginning of the computational graph, until this variable. For instance
        x=Variable([1,2,-12])
        
        x.gradient is equal to Identity(3)
        
        y=ad.sin(x)
        
        y.gradient is the square matrix with diagonal elements (cos(1), cos(2), cos(-12))
        
        z=x.power(2)
        
        z.gradient is the square matrix with diagonal elements of (2*cos(1)*1, 2*cos(2)*1, 2*cos(-12)*1)
        the *1 term corresponds to the gradient of the variable x
    """
    def __init__(self,data, gradient=None):
    
             
        #converts list or float to numpy array
        self.data=utils.data_2_numpy(data)
        
        #get the shape of the data provided, is it a float or a vector
        data_shape=utils.get_shape(data)
        
        
        #check if a gradient is provided, or if we initialize a Variable from scratch
        if type(gradient) != type(None):
            self.gradient=utils.data_2_numpy(gradient)    
        
        else:
            #default value for the gradient
            
            # for now we only handle AD on scalar values and 1-D vectors
            if len(data_shape)>2:
                print('dealing with high order tensors as input. not handled yet')
                raise NotImplementedError
            
            if len(data_shape) == 2 :
                if data_shape[0]!=data_shape[1]:
                    print('dealing with non square matrices, not handled yet')
                    raise NotImplementedError
                    
            #get the number of dimensions
            lenght=data_shape[0]
            
            #default value for the gradient, which is actually a Jacobian because we deal with 
            #vectorial functions
            
            self.gradient = np.eye(lenght)
        
    def set_data(self, data):
        """
        set the data of a Variable. the new data needs to have the same shape as the previous data
        stored in this variable
        """
        
        new_shape= utils.get_shape(data)
        assert new_shape==self.data.shape, 'trying to set data with inconsistent shapes! previous shape : {} -- shape provided {}'.format(self.data.shape, new_shape)
        self.data=data
    
    def set_gradient(self, gradient):
        """
        set the gradient of a Variable. the new gradient needs to have the same shape as the previous
        gradient stored in this variable
        """
        new_shape= utils.get_shape(gradient)
        assert new_shape==self.gradient.shape, 'trying to set data with inconsistent shapes! previous shape : {} -- shape provided {}'.format(self.gradient.shape, new_shape)
        
        self.gradient=gradient
    
    def __str__(self):
        """
        nice print to see what is in the variable
        """        
        return('data : {} \ngrad : {}'.format(self.data, self.gradient))
       
    def __scalar_to_variable(self, other):
        const_vec = [other]*self.data.shape[0]
        return Variable(const_vec, gradient=np.zeros(self.data.shape))
   
    def __add__(self, other):
        """
        overload addition
        """
        
        if not 'add' in dir():
            from autograd.blocks.operations import add
            add=add()
        
        if not isinstance(other, Variable):
            other=self.__scalar_to_variable(other)
        return add(self, other)

    def __radd__(self, other):
        """
        overload right-addition
        """
        
        if not 'add' in dir():
            from autograd.blocks.operations import add
            add=add()
            
            
        if not isinstance(other, Variable):
            other=self.__scalar_to_variable(other)
        return add(self, other)

    def __sub__(self, other):
        """
        overload subtraction
        """
        
        if not 'substract' in dir():
            from autograd.blocks.operations import subtract
            subtract=subtract()
            
            
        if not isinstance(other, Variable):
            other=self.__scalar_to_variable(other)
        return subtract(self, other)

    def __rsub__(self, other):
        """
        overload right-subtraction (order matters)
        """
        
        if not 'substract' in dir():
            from autograd.blocks.operations import subtract
            subtract=subtract()
            
            
        if not isinstance(other, Variable):
            other=self.__scalar_to_variable(other)
        return subtract(other, self)

    def __mul__(self, other):
        """
        overload element-wise multiplication
        """
        
        if not 'multiply' in dir():
            from autograd.blocks.operations import multiply
            multiply=multiply()
            
            
        if not isinstance(other, Variable):
            other=self.__scalar_to_variable(other)
        return multiply(self, other)

    def __rmul__(self, other):
        """
        overload element-wise multiplication
        """
        if not 'multiply' in dir():
            from autograd.blocks.operations import multiply
            multiply=multiply()
            
        if not isinstance(other, Variable):
            other=self.__scalar_to_variable(other)
        return multiply(other, self)

    def __truediv__(self, other):
        """
        overload division
        """
        
        if not 'divide' in dir():
            from autograd.blocks.operations import divide
            divide=divide()
            
            
        if not isinstance(other, Variable):
            other=self.__scalar_to_variable(other)
        return divide(self, other)

    def __rtruediv__(self, other):
        """
        overload right-division (order matters)
        """
        if not 'divide' in dir():
            from autograd.blocks.operations import divide
            divide=divide()
            
        if not isinstance(other, Variable):
            other=self.__scalar_to_variable(other)
        return divide(other, self)

    def __pow__(self, other):
        """
        overload power
        """
        
        if not 'power' in dir():
            from autograd.blocks.operations import power
            power=power()
            
            
        if isinstance(other, Variable):
            raise ValueError('Power is not supported for type Variable')
        return power(self, other)
       
        
    def __neg__(self):
        """
        implementing the - unary overloading
        """        
        return Variable(-self.data, -self.gradient)
    

    def __getitem__(self, key):
        """
        overload extracting elements from a vector
        works for both integer and slice
        """
        new_data=self.data[key]       
        number_of_dimensions_in_this_variable = self.gradient.shape[1]        
        new_grad=self.gradient[key,:].reshape(-1,number_of_dimensions_in_this_variable)
        
        return (Variable(new_data, new_grad))

    def __eq__(self,other):
        """
        overload equals dunder method
        """
        if self.data == other.data and self.gradient == other.gradient:
            return True
        else:
            return False

    def __ne__(self,other):
        """
        overload not equal dunder method
        """
        if self.data != other.data or self.gradient != other.gradient:
            return True
        else:
            return False

        
if __name__=='__main__':
    
    #small local test, should be put in a dedicated file in the test folder
    x=Variable([2,3,4])
    
    print(x.data)
    print(x.gradient)
    
    new_data=np.ones(3)
    x.set_data(new_data)
    print(x.data)
    
    new_grad=43*np.eye(3)
    x.set_gradient(new_grad)
    print(x)
    
    print('==== Operators ====')
    
# =============================================================================
    print(x+3)
    print(3-x)
    print(x/3)
    print(3*x)
    print(x+x)
    print(x-x)
    print(x*x)
    print(x/x)
    print(x**2)


# =============================================================================
