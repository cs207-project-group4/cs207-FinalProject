# -*- coding: utf-8 -*-
# =============================================================================
# Contains optimization functions for the AD package
# =============================================================================


class Optimizer():
    """

    Optimizer Base Class

    == Args ==

    loss_func (function): a function accepting a list of `params` (see below) and returning a tuple (data, gradient)
    params (array): an array of initialization parameters - these should correspond to the parameters of the loss_func
    lr (float): leaning rate for steps
    tol (float): tolerance for determining loss function convergence
    max_iter (int): maximumer number of steps the optimizer will run

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
        """

        loop until tolerance is met (convergence) or for max number of iters

        """
        count = 0
        while count < self.max_iter:

            prev_loss, prev_grad = self.loss_func(self.params)
            self.step()
            new_loss, new_grad = self.loss_func(self.params)

            if abs(prev_loss - new_loss) < self.tol:
                break
            count += 1

        return(self.params)


class GD(Optimizer):

    """
    Gradient Descent Optimizer

    Example:
    >>> import numpy as np
    >>> import autograd as ad
    >>> from autograd.variable import Variable
    >>> from autograd.optimize import GD
    >>> def loss(params):
    >>>     var = Variable(params)
    >>>     x = var[0]
    >>>     y = var[1]
    >>>     l = (x+5)**2 + (y+3)**2
    >>>     return (l.data, l.gradient)
    >>> x_init = [10, 4]
    >>> optimize_GD = GD(loss, x_init, lr=0.01, max_iter=100000, tol=1e-18)
    >>> optimize_GD.solve()
    >>> array([-5. -3.])
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        # pass current params into loss_func
        loss, grad = self.loss_func(self.params)
        grad = grad[0]

        # update params with a gradient step
        self.params = self.params - self.lr * grad


class SGD(Optimizer):

    def __init__(self):
        pass
