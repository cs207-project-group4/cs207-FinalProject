autograd Usage
==============

Additional resources are available in Demo_Notebook.ipynb - make sure to have matplotlib installed if you want to run the Demo_Notebook

Simple Differentiation
------------------------

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


Forward or Reverse Mode
-----------------------

Forward mode is set by default, but to explicitly set forward mode::

    >>> import autograd as ad
    >>> ad.set_mode('forward')

Reverse mode can be easily set::

    >>> ad.set_mode('reverse')

Once reverse mode is set, all differentiation in the session will be calculated in reverse mode, unless forward mode is explicitly set


Optimization
------------

Currently, autograd supports gradient descent and Adam optimization, in both forward and reverse mode

Gradient Descent
----------------

Gradient Descent Optimization::

    import numpy as np
    import autograd as ad
    from autograd.variable import Variable
    from autograd.optimize import GD

    #set to forward mode
    ad.set_mode('forward')

    #define function
    def loss(params):
        var = Variable(params)
        x,y = var[0], var[1]
        l = (x+5)**2 + (y+3)**2

        l.compute_gradients()

        return (l.data, l.gradient)
    #initialize values
    x_init = [10, 4]
    #create optimization object and set parameters
    optimize_GD = GD(loss, x_init, lr=0.1, max_iter=1000, tol=1e-13)
    #solve
    sol = optimize_GD.solve()
