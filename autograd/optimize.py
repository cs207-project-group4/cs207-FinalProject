# -*- coding: utf-8 -*-
# =============================================================================
# Contains optimization functions for the AD package
# =============================================================================
import numpy as np

class Optimizer():
    """
    Optimizer Base Class

    == Args ==

    loss_func (function): a function accepting a list of `params` (see below) and returning a tuple (data, gradient)
    params (array): a list of initialization parameters - these should correspond to the parameters of the loss_func
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

    def solve(self, return_steps=False):
        """
        Loop until convergence criteria is met or for max_iters
        """
        steps=[]
        count = 0
        while count < self.max_iter:

            prev_loss, prev_grad = self.loss_func(self.params)
            self.step()
            new_loss, new_grad = self.loss_func(self.params)
            if return_steps:
                steps.append(self.params)

            if abs(prev_loss - new_loss) < self.tol:
                break
            count += 1

        if return_steps:
            return (self.params, steps)
        else:
            return (self.params)


class GD(Optimizer):
    """
    Gradient Descent Optimizer

    Example:
    >>> import numpy as np
    >>> import autograd as ad
    >>> from autograd.variable import Variable
    >>> from autograd.optimize import GD
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        # pass current params into loss_func
        loss, grad = self.loss_func(self.params)
        grad = grad[0]

        # update params with a gradient step
        self.params = self.params - self.lr * grad


class Adam(Optimizer):
    """
    Implements Adam Optimizer (`Adam: A Method for Stochastic Optimization`)
    """

    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.exp_avg=np.zeros_like(self.params)
        self.exp_avg_sq=np.zeros_like(self.params)
        self.step_count=0

    def step(self):
        # increment step count
        self.step_count += 1

        # get current loss
        loss, grad = self.loss_func(self.params)
        grad = grad[0]

        # calculate moving averages
        self.exp_avg = self.exp_avg*self.beta1 + (1-self.beta1)*grad
        self.exp_avg_sq = self.exp_avg_sq*self.beta2 + (1-self.beta2)*(grad**2)

        # perform bias correction
        bias_correction1 = self.exp_avg / (1 - self.beta1 ** self.step_count)
        bias_correction2 = self.exp_avg_sq / (1 - self.beta2 ** self.step_count)

        # calculate step size and perform step
        step_size = self.lr * bias_correction1 / (np.sqrt(bias_correction2) + self.eps)
        self.params = self.params - step_size
