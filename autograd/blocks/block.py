# -*- coding: utf-8 -*-\
import numpy as np
from autograd.variable import Variable



# =============================================================================
# IMPORTANT (MATHEMATICAL) NOTE
# what is named here as gradient, is a wrong denomination
# as we are dealing with vector function, the right word to use is more Jacobian
# =============================================================================

class Block():
    """
    main class for the blocks of the AD package. The several blocks will be of several types
    For now we envision two types of block : simple and double, depending on the number of 
    variables as input.
    For instance : 
        sin(.) is a simple block as it has only one input
        
        dot(.,.) is a double block as it has two inputs
        
        
    NOTE : Maybe we should define the following functions as class methods
    This way we will not have to instantiate these class to use them.
    for instance instead of doing : 
        import sin
        sinBlock = sin()
        y=sinBlock(x)
        
    we could have an implementation like : 
        import sin
        y=sin(x)
    """
    
    
    def data_fn(self,*args):
        """
        function to apply to the input Variable.
        for instance :
            sin.data_fn(x) will return sin(x)
        """
        raise NotImplementedError
    
    def gradient_fn(self, *args):
        """
        function implementing the gradient of data_fn.
        for instance : 
            sin.gradient_fn(x) will return cos(x)
        """
        raise NotImplementedError
        
    def gradient_forward(self, *args):
        """
        function implementing the forward pass of the gradient.
        this function will depend on wether this is a simple block or a double block.
        for instance :
            sin.gradient_forward(x) will return grad(x) * cos(x)
            
            multiply(x,y) will return : grad(x)*y + x*grad(y)
        """
        raise NotImplementedError
    
    def __call__(self, *args):
        """ 
        applies the forward pass of the data and the gradient.
        returns a new variable with the updated information on data and gradient.
        """ 
        new_data=self.data_fn(*args)
        new_grad=self.gradient_forward(*args)
        
        
        return(Variable(new_data, new_grad))    
        
    
class SimpleBlock(Block):
    """
    this block is meant to implement one-variable functions that have been vectorized : 
        sin(vector) is the vector of coordinates (sin(vector[i])) 
        
    For this, these functions are functions from Rn to Rn and they have a Jacobian which is
    a square matrix, with elements only on the diagonal
    """
    def gradient_forward(self, *args):
        
        assert len(args)==1, 'This type of block takes only one input, {} were given'.format(len(args))
        
        simpleInput=args[0]
        
        #get the elements of the diagonal
        elts = self.gradient_fn(simpleInput)
        jacobian = np.diag(elts)
        new_grad = np.dot(jacobian, simpleInput.gradient)
        
        return(new_grad)


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
        