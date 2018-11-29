# -*- coding: utf-8 -*-
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
    >>>     b1 = (x+5)**2
    >>>     return(b1)
    >>> optimize_GD = GD(params = [1], lr = 0.01,tolerance=0.00001,max_iter = 10000)
    >>> optimize_GD.solve(function)
    array([-4.99951078])


    """
    def __init__(self,params,lr,tolerance,max_iter = 10000):
        self.params = params
        self.lr  = lr
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.delta = 0


    def step(self,function):
        #return one step in the block

        #find the gradient of the function
        block = function(self.params)

        #compute the delta
        self.delta = self.lr * block.gradient

        #upate the params
        new_params = self.params - self.delta
        self.params = new_params[0]


    def solve(self,function):

        #loop until tolerance or max number of iters is met
        count = 0
        while count < self.max_iter:
            self.step(function)
            if all(abs(i) <= self.tolerance for i in self.delta[0]):
            #if abs(self.delta) < self.tolerance:
                break
            count += 1

        #return final updated parameters
        return(self.params)
