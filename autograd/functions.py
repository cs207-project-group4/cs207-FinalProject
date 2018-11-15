from autograd.blocks.hyperbolic import sinh, cosh, tanh
from autograd.blocks.operations import add, subtract, multiply, divide, power
from autograd.blocks.trigo import sin, cos, tan
from autograd.blocks.expo import exp, log

sinh_=sinh()
cosh_=cosh()
tanh_=tanh()
add_=add()
subtract_=subtract()
multiply_=multiply()
divide_=divide()
power_=power()
sin_=sin()
cos_=cos()
tan_=tan()
exp_=exp()
log_=log()

def sinh(x):
	return sinh_(x)

def cosh(x):
	return cosh_(x)