autograd Usage
==============

Additional resources are available in Demo_Notebook.ipynb - make sure to have matplotlib installed if you want to run the Demo_Notebook

Forward Mode Differentiation
------------------------------------

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
