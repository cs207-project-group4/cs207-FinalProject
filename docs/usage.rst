Autograd Usage
==============

`Autograd` comes with an user-friendly API, for both forward and reverse mode.

General rules
-------------

Autograd will nearly always give you a result. However, in order to ensure that you compute what you exactly think you are computing, please make sure to read carefully these points : 

1. When you define a Variable, it is automatically set as the input node of the computational graph

2. Thus, if you define two variables like ``x=Variable(3)`` and then ``y=Variable(4)``, the input node of the graph will be y only, and you will not compute gradients with respect to x. Never.

3. If you want to work on function of several inputs, please refer to the section `Multiple Inputs`

4. Before you try to access the ``variable.gradient`` attribute, you should run ``variable.compute_gradients().

5. When you are in reverse mode, don't forget to reset the computational graph when you are running a new function call. You can do it with ``ad.reset_graph()``

6. Enjoy! :)




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



As this package handles vector to vector mapping, we can theoretically consider every function of several variables as a function of vector input. For exemple, we can see the function f(x,y,z) as a function of 3 variables which are scalar, but also as a function of one variable, which is a vector of R3. We refer to these methods to the Multiple Variables Mode and Vector Mode, respectively.


Vector Mode
^^^^^^^^^^^^

Before performing any operations, you should embbed the inputs of your function in one big variable ::

    def vector_function(x,y):
        big_variable = Variable([x,y])   
        x,y=big_variable[0], big_variable[1]

        b1 = ad.exp(-0.1*((x**2)+(y**2)))
        b2 = ad.cos(0.5*(x+y))
        b3 = b1*b2+0.1*(x+y)+ad.exp(0.1*(3-(x+y)))


        b3.compute_gradients()
        return(b3.data,b3.gradient)
         
In that case, you will have `b3.gradient` as a matrix of shape 1*3, because you considered the function as a vector function mapping from R3 to R.


Multiple Variables
^^^^^^^^^^^^^^^^

    def vector_function(x,y):
         x,y=av.Variable.multi_variables(x,y)

         b1 = ad.exp(-0.1*((x**2)+(y**2)))
         b2 = ad.cos(0.5*(x+y))
         b3 = b1*b2+0.1*(x+y)+ad.exp(0.1*(3-(x+y)))


         b3.compute_gradients()
         return(b3.data,b3.gradient)


In that case, we have ``b3.gradient = [grad(b3, x), grad(b3, y)]``  with ``grad(b3,x)`` refers to the gradient of the function ``x-->b3`` evaluated at x.


The choice of which mode is up to you, the multi_variables is useful when you deal with several inputs with different shapes : 

def vector_function(x,L,N):
         x, L, N = av.Variable.multi_variables(x,L, N)

         b1 = ad.sum_elts(L)
         b2=x*L
         b3=x+b2
         b4=N*L
         b5=b3+b4[0]
         
         b5.compute_gradients()
         return(b5.data,b5.gradient)
      
We will then have ``b5.gradient = [grad(b5,x), grad(b5,L), grad(b5,L)]`` with ``grad(b5, L)`` a matrix of shape 1*dim(L), etc...

This method is quite straightforward and intuitive, not as what we would have had to do in the vector mode to get the gradients of x and L separately :: 

    gradient_b5_x = b5.gradient[:,0:1]
    gradient_b5_L = b5.gradient[:,1:dim(L)+1]
    gradient_b5_N = b5.gradient[:,dim(L)+1:]
    
with even more complicated gradient extractions when you have more input vectors of different sizes...


The performance of these two methods is identical.

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
