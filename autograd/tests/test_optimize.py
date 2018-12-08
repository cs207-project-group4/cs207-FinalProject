import numpy as np
import autograd as ad
from autograd.variable import Variable
from autograd.optimize import GD


def test_gradient_descent():

    def loss(params):
        var = Variable(params)
        x = var[0]
        y = var[1]
        l = (x+5)**2 + (y+3)**2
        return (l.data, l.gradient)

    x_init = [10, 4]
    optimize_GD = GD(loss, x_init, lr=0.01, max_iter=100000, tol=1e-18)
    sol = optimize_GD.solve()
    assert round(sol[0]) == -5 and round(sol[1]) == -3
