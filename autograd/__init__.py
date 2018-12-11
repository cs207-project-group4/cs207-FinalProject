# -*- coding: utf-8 -*-

"""

We instantiate all of our blocks in the __init__ so that we can
call them in the following fashion:

>>> import autograd as ad
>>> from autograd.variable import Variable
>>> x = Variable(3)
>>> y = ad.sinh(x)

"""
import numpy as np
from autograd.blocks.hyperbolic import sinh, cosh, tanh
from autograd.blocks.operations import add, subtract, multiply, divide, power, sum_elts
from autograd.blocks.trigo import sin, cos, tan, arcsin, arccos, arctan
from autograd.blocks.expo import exp, log, sqrt
from autograd.node import C_graph
from autograd import config

#default mode for computing gradients
mode='forward'

#computational graph for reverse mode
c_graph=C_graph()

def reset_graph():
    c_graph.reset_graph()



def set_mode(new_mode):
    global mode, c_graph
    mode=new_mode

    if new_mode=='reverse':
        reset_graph()

# =============================================================================
# shortcuts for better user interface
# =============================================================================

sin_=sin()
cos_=cos()
tan_=tan()
exp_=exp()
log_=log()
sqrt_=sqrt()
sinh_=sinh()
cosh_=cosh()
tanh_=tanh()
arcsin_=arcsin()
arccos_=arccos()
arctan_=arctan()
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

def log(x, base=np.e):
	return log_(x, base=base)

def sqrt(x):
	return sqrt(x)

def sinh(x):
	return sinh_(x)

def cosh(x):
	return cosh_(x)

def tanh(x):
	return tanh_(x)

def arcsin(x):
	return arcsin_(x)

def arccos(x):
	return arccos_(x)

def arctan(x):
	return arctan_(x)

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
