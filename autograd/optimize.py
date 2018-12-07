# -*- coding: utf-8 -*-
# =============================================================================
# Contains optimization functions for the AD package
# =============================================================================


class Optimizer():
    """

    Optimizer Base Class

    """
    def __init__(self, loss_func, params, lr=0.01, max_iter=100000, tol=1e-14):
        self.loss_func = loss_func
        self.params = params
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def step(self):
        """

        Performs a single step of the optimizaiton

        """
        raise NotImplementedError

    def solve(self):
        #loop until tolerance is met (convergence) or max number of iters is met
        count = 0
        while count < self.max_iter:
            prev_loss = self.loss_func(self.params)
            self.step()
            if abs(prev_loss - self.loss_func(self.params)) < self.tol:
                break
            count += 1

        #return final params
        return(self.params)




class GD(Optimizer):

    """
    Gradient Descent Optimizer

    Init Arguments:
    params (array): an array of initialization parameters - these should correspond to the parameters of the function, parameters are position specific
    lr (float): leaning rate
    tolerance (float): gradient descent tolerance
    max_iter (int): maximumer number of steps the gradient descent solver will run


    Example:
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        # pass current params into loss_func
        loss = self.loss_func(self.params)

        # update params with a gradient step
        self.params = self.params - self.lr * loss.gradient


class SGD(Optimizer):

    def __init__(self):
        pass
