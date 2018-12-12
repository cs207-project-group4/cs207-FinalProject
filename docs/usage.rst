Autograd Usage
==============

Additional resources are available in Demo_Notebook.ipynb - make sure to have matplotlib installed if you want to run the Demo_Notebook

Simple Differentiation Case
----------------------------

Example: How to differentiate ``f(x) = sin(x) + cos(x)`` at x = pi::

    >>> import numpy as np
    >>> import autograd as ad
    >>> from autograd.variable import Variable
    >>> ad.set_mode('forward')
    >>> x = Variable(np.pi)
    >>> b1 = ad.sin(x)
    >>> b2 = ad.cos(x)
    >>> b3 = b1 + b2
    >>> print(b3.gradient)
    -1

b3 will contain the gradient of ``y = sin(x) + cos(x)`` at x = pi

Example: How to differentiate ``f(x)=sin(cos(x+3)) + e^(sin(x)^2)`` at x = 1::

    >>> import numpy as np
    >>> import autograd as ad
    >>> from autograd.variable import Variable
    >>> ad.set_mode('forward')
    >>> x = Variable(1)
    >>> b1 = x + 3
    >>> b2 = ad.sin(x)
    >>> b3 = ad.cos(b1)
    >>> b4 = ad.sin(b3)
    >>> b5 = b2*b2
    >>> b6 = ad.exp(b5)
    >>> b7 = b6 + b4
    >>> print(b7.gradient)
    2.44674864

b7 will contain the gradient of ``f(x)=sin(cos(x+3)) + e^(sin(x)^2)`` at x = 1



Differentiation of Functions
-----------------------------

If a user wants to differentiate multiple values is recommended that users create functions that wrap around autograd::

    def function(x):
        x1 = av.Variable(x)
        b1 = ad.sin(x1)
        b2 = ad.cos(x1)
        b3 = b1 + b2
        b3.compute_gradients()
        return(b3.data,b3.gradient)

This function can be used to loop and differentiate values::

    value = list()
    data = list()
    gradient = list()
    for i in np.linspace(-2 * np.pi, 2 * np.pi):
        value.append(i)
        output = function(i)
        data.append(output[0])
        gradient.append(output[1][0])


Multiple Inputs
----------------

Many applications of automatic differentiation require the use of multiple inputs. In order to add multiple input variables, use the ``multi_variables`` method of the ``Variable`` class::

    def vector_function(x,y):
         x,y=av.Variable.multi_variables(x,y)

         b1 = ad.exp(-0.1*((x**2)+(y**2)))
         b2 = ad.cos(0.5*(x+y))
         b3 = b1*b2+0.1*(x+y)+ad.exp(0.1*(3-(x+y)))


         b3.compute_gradients()
         return(b3.data,b3.gradient)


In case of multiple inputs, the ``.gradient()`` method returns the gradients of the output node with respect to each of the inputs



Forward or Reverse Mode
-----------------------

Forward mode is set by default, but to explicitly set forward mode::

    >>> import autograd as ad
    >>> ad.set_mode('forward')

Reverse mode can be easily set::

    >>> import autograd as ad
    >>> ad.set_mode('reverse')

Once reverse mode is set, all differentiation in the session will be calculated in reverse mode, unless forward mode is explicitly set.::

    >>> import autograd as ad
    >>> ad.set_mode('reverse')
    >>> ad.set_mode('forward')

The resulting setting it forward mode


Optimization
------------

Currently, autograd supports gradient descent and Adam optimization, in both forward and reverse mode.

Optimization Setup::

    import numpy as np
    import autograd as ad
    from autograd.variable import Variable

    #set to forward mode
    ad.set_mode('forward')

    #define function
    def loss(params):
        var = Variable(params)
        x,y = var[0], var[1]
        l = (x+5)**2 + (y+3)**2

        l.compute_gradients()

        return (l.data, l.gradient)


Gradient Descent
----------------
Autograd has implemented `Gradient Descent <https://en.wikipedia.org/wiki/Gradient_descent>`_.

Gradient Descent Optimization::

    #import gradient descent
    from autograd.optimize import GD

    #initialize values
    x_init = [10, 4]

    #create optimization object and set parameters
    optimize_GD = GD(loss, x_init, lr=0.1, max_iter=1000, tol=1e-13)

    #solve
    sol = optimize_GD.solve()

Adam
-----
Autograd has implemented the Adam Optimizer: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

Adam Optimization::

    #import Adam Optimizer
    from autograd.optimize import Adam

    #initialize values
    x_init = [10, 4]

    #create optimization object and set parameters
    adam = Adam(loss, x_init, lr=0.1, max_iter=1000, tol=1e-13)

    #solve
    sol = adam.solve()
