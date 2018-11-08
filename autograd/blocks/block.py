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
        function implementing the gradient of data_fn, when it is easy to express
        for instance : 
            sin.gradient_fn(x) will return cos(x)
        """
        raise NotImplementedError
        
    def gradient_forward(self, *args):
        """
        function implementing the forward pass of the gradient.
        
        Let's consider a computational graph which transforms
        x_0 --> x_1 --> x_2 --> x_3 --> y
        let's call the output of that block y, then the output of 
        gradient_forward(x3), will contain the jacobian of the function x_0 --> y
    
        this function is in charge of pushing the gradients forward, it will combine the
        previously computed gradients to the derivative of this block_function
        
        
        
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


