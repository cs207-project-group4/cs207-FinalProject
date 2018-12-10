import numpy as np
import autograd as ad
from autograd.variable import Variable
from autograd.optimize import GD, Adam


def test_gradient_descent_forward():
    ad.set_mode('forward')

    def loss(params):
        var = Variable(params)
        x,y = var[0], var[1]
        l = (x+5)**2 + (y+3)**2

        l.compute_gradients()

        return (l.data, l.gradient)

    x_init = [10, 4]
    optimize_GD = GD(loss, x_init, lr=0.1, max_iter=1000, tol=1e-13)
    sol = optimize_GD.solve()
    assert round(sol[0]) == -5 and round(sol[1]) == -3


def test_gradient_descent_reverse():
    ad.set_mode('reverse')

    def loss(params):
        ad.reset_graph()
        var = Variable(params)
        x,y = var[0], var[1]
        l = (x+5)**2 + (y+3)**2

        l.compute_gradients()

        return (l.data, l.gradient)

    x_init = [10, 4]
    optimize_GD = GD(loss, x_init, lr=0.1, max_iter=1000, tol=1e-13)
    sol = optimize_GD.solve()
    assert round(sol[0]) == -5 and round(sol[1]) == -3


def test_adam_forward():
    ad.set_mode('forward')

    def loss(params):
        var = Variable(params)
        x = var[0]
        y = var[1]
        l = (x+5)**2 + (y+3)**2

        l.compute_gradients()
        return (l.data, l.gradient)

    x_init = [10, 4]
    adam = Adam(loss, x_init, lr=0.1, max_iter=1000, tol=1e-13)
    sol = adam.solve()
    assert round(sol[0]) == -5 and round(sol[1]) == -3


def test_adam_reverse():
   ad.set_mode('reverse')

   def loss(params):
       ad.reset_graph()
       var = Variable(params)
       x = var[0]
       y = var[1]
       l = (x+5)**2 + (y+3)**2

       l.compute_gradients()
       return (l.data, l.gradient)

   x_init = [10, 4]
   adam = Adam(loss, x_init, lr=0.5, max_iter=1000, tol=1e-13)
   sol = adam.solve()
   assert round(sol[0]) == -5 and round(sol[1]) == -3
