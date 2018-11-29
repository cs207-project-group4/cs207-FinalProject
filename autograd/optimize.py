
# =============================================================================
# Contains optimization functions for the AD package
# =============================================================================


class Optimize():
    """

    Optimizer Base Class

    """
    def __init__(self,learning_rate, loss_function):
        self.learning_rate = learning_rate
        self.loss_function = loss_function



    def step(self, **args):
        """

        Performs a single step of the optimizaiton

        """
        raise NotImplementedError



class GD(Optimize):

    """
    Gradient Descent Optimizer

    Arguments:

    function (block class): this is the final

    params (iterable): a dictionary of initialization parameters - dictionary keys should match

    lr (float): leaning rate


    >>> import numpy as np
    >>> import autograd as ad
    >>> from autograd.variable import Variable
    >>> from autograd.optimize import GD
    >>> def function(x_0):
    >>>     x = Variable(x_0)
    >>>     b1 = ad.sin(x)
    >>>     b2 = ad.cos(x)
    >>>     b3 = b1 + b2
    >>>     return(b3)
    >>> optimize_GD = GD(params = [1], lr = 0.01,tolerance=1,n_steps = 1000)
    >>> optimize_GD.solve(function)
    array([-2.3561913])


    """
    def __init__(self,params,lr,tolerance,n_steps):
        self.params = params
        self.lr  = lr
        self.tolerance = tolerance
        self.n_steps = n_steps


    def step(self,function):
        #return one step in the block

        #find the gradient of the function
        block = function(self.params)

        #upate the params
        new_params = self.params - (self.lr * block.gradient)
        self.params = new_params[0]


    def solve(self,function):

        #perform n number of steps
        for i in range(self.n_steps):
            self.step(function)

        #return most updated parameters
        return(self.params)
