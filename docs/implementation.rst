
Implementation
==============

Recalling the `background section <https://autograd.readthedocs.io/en/latest/background.html>`_, we saw that the automatic differentiation framework splits a complex function into several atomic functions which derivative is easy to compute. Then, the results are aggregated using the chaing rule. 

This package has been designed so that it is easy for a new user to define his own new atomic function. For instance, we did not implement convolution operations over vectors, but a new user could easily define it, following the API we will describe.

**Important :** The functionment of the package is slightly different depending on wether you use the forward or the reverse mode. In this optic, we will first present the forward mode and then highlight the differences happening in the reverse mode.


Forward Mode
------------

The core data structures in this package are ``Variables`` and ``Blocks``.

We are going to consider that every function can be split into core atomic functions, each of which we will call a `Block`. Thus, the application of a function is a mere composition of `Block` operations to `Variable`s. 

.. image:: img/basic_function.png

**Variable**

The first core data structure is `Variable`. This object will flow through several `Blocks`, storing the new values of the functions computed, as well as the gradient computed so far.

.. image:: img/Variable.png

It contains two main attributes : ``data`` and ``gradient``. In each block, the input ``Variable`` brings the information from the previous functions and gradients computed and propagates the data and gradient flow forward. Note that because our package deals with vector functions, the ``gradient`` attribute is actually a ``Jacobian`` matrix.

If nothing is indicated by the user, the default value of ``Variable.gradient`` is an Identity matrix, meaning we are at the beginning of the computational graph.

For now, the constants are managed as Variables with a initial ``Jacobian`` as a matrix of 0. It is not efficient in the way that we still use this matrix of 0 for the gradient flow, we will probably optimize it at the next iteration.

**Block**

The second core data structure is the ``Block``. It is an atomic operation performed on ``Variable``. For instance, sin, exp, addition or multiplication.

.. image:: img/Block.png

The ``Block`` contains two major methods : ```data_fn ``` and ```gradient_fn ```.

```data_fn ``` is used to compute the function evaluation for that block. For example we can use::

    import autograd as ad
    from autograd.variable import Variable

    #instantiate a block
    x= Variable(3)
    y= ad.sin(x)

and the new ``Variable`` y, will have its ``data`` attribute set to ``av.trig.sin.data_fn(3)`` = ``sin(3)``

``gradient_fn`` is used to compute the gradient evaluation for that block. Keeping the same example, we have::

    import autograd as ad
    from autograd.variable import Variable
    #instantiate a block
    x= Variable(3)
    y= ad.sin(x)

As previously stated, the variable x has the default value for ``gradient``, which is an array of ones. Then, the block sin will create a new variable y, which ``data`` attribute has already been explained above. The ``gradient`` attribute is set to ``ad.block.sin.gradient_fn(3) * x.gradient = cos(3) * 1``

Note that for more complex functions, the ``gradient_fn`` is combined with the method ``gradient_forward``. For the multiplication for instance, we will use ``gradient_forward`` to push forward the gradient flow, same for the addition, and other basic operations.

The way to see ``gradient_forward`` is the following :
Let's consider a computational graph which transforms : x_0 --> x_1 --> x_2 --> x_3 --> y

let's call the output of the last block y, then the output of gradient_forward(x_3), will contain the jacobian of the function x_0 --> y. More generally, the output of gradient_forward(x_i) will contain the Jacobian matrix of the function : x_0 --> x_i

this function is in charge of pushing the gradients forward, it will combine the previously computed gradients to the derivative of this block_function

*No storing of the computational graph*

The solution we provided is efficient in that we don't store the computation graph. The values of the variables are computed on the fly, both data and gradient.

*Classes implemented*

As hinted before, we will have a class for the `Variable` and another class for `Block`.
Though each elementary function will be assigned a subclass of `Block` : we will have a set of `Block` functions hard-coded from which we expect the user to build his/her complicated combinations.

Example of this set could be: sin, cos, tan, exp, pow, sum, mean, ...

Of course, the ``autograd`` package is being built respecting the design patterns for good development, the user will have the possibility to build his own `Block` if he would not find a specific function among the ones we provide. The user would have to follow the `Block` interface and provide a ``data_fn`` as well as a ``grad_fn`` (leveraging *duck typing*).

*External dependencies*

The package is highly reliant on ``Numpy``. The Demo_Notebook uses ``matplotlib``, but ``matplotlib`` is not required for the autograd to run. 
