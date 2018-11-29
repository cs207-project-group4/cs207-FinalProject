import numpy as np
import autograd as ad
from autograd.variable import Variable
from autograd.optimize import GD


def test_gradient_descent():

    def function(x_0):
        x = Variable(x_0)
        b1 = (x+5)**2
        return b1

    optimize_GD = GD(params = [10], lr = 0.01,tolerance=0.0001,max_iter = 10000)
    assert round(optimize_GD.solve(function)[0]) == -5
