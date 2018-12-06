# -*- coding: utf-8 -*-

"""

We instantiate all of our blocks in the __init__ so that we can
call them in the following fashion:

>>> import autograd as ad
>>> from autograd.variable import Variable
>>> x = Variable(3)
>>> y = ad.sinh(x)

"""

from autograd.blocks.hyperbolic import sinh, cosh, tanh
from autograd.blocks.operations import add, subtract, multiply, divide, power, sum_elts
from autograd.blocks.trigo import sin, cos, tan
from autograd.blocks.expo import exp, log


#default mode for computing gradients
mode='forward'

#ids used for the nodes in the computational graph
ids=[]

sin_=sin()
cos_=cos()
tan_=tan()
exp_=exp()
log_=log()
sinh_=sinh()
cosh_=cosh()
tanh_=tanh()
add_=add()
subtract_=subtract()
multiply_=multiply()
divide_=divide()
power_=power()
sum_elts_=sum_elts()

# ================
#    FUNCTIONS
# ================

def sin(x):
	return sin_(x)

def cos(x):
	return cos_(x)

def tan(x):
	return tan_(x)

def exp(x):
	return exp_(x)

def log(x):
	return log_(x)

def sinh(x):
	return sinh_(x)

def cosh(x):
	return cosh_(x)

def tanh(x):
	return tanh_(x)

# ================
#	 OPERATORS
# ================

def add(x, y):
	return add_(x,y)

def subtract(x, y):
	return subtract_(x, y)

def multiply(x, y):
	return multiply_(x, y)

def divide(x, y):
	return divide_(x, y)

def power(x, y):
	return power_(x, y)

def sum_elts(x):
	return sum_elts_(x)
