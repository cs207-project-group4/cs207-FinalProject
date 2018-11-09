from autograd.blocks.operations import add
from autograd.blocks.operations import subtract
from autograd.blocks.operations import multiply
from autograd.blocks.operations import divide
from autograd.blocks.operations import power

from autograd.variable import Variable
import numpy as np

def test_add_over():
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    x=Variable(datax)
    y=Variable(datay)

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block= x + y
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.add(datax,datay)
    gradient_true=2*np.identity(5)
# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

def test_subtract_over():
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    x=Variable(datax)
    y=Variable(datay)
# =============================================================================
#   define custom block
# =============================================================================
    subtract_block=subtract()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block= x - y
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.subtract(datax,datay)
    gradient_true=np.zeros((5,5))
# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sub data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sub gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

def test_multiply_over():
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    x=Variable(datax)
    y=Variable(datay)
# =============================================================================
#   define custom block
# =============================================================================
    multiply_block=multiply()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block= x*y

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.multiply(datax,datay)
    gradient_true=np.identity(5)*y.data + np.identity(5)*x.data
# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong mult data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong mult gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

def test_divide_over():
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    x=Variable(datax)
    y=Variable(datay)
# =============================================================================
#   define custom block
# =============================================================================
    divide_block=divide()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block= x/y
# =============================================================================
#   define expected output
# =============================================================================
    data_true =np.divide(datax,datay)

    gradient_true = (x.gradient*y.data-x.data*y.gradient)/y.data**2
# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong div data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass, use allclose because division create small differences.  Outputs look same though!
# =============================================================================
    assert np.allclose(gradient_true, y_block.gradient, rtol = .001), 'wrong div gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)


def test_power_over():
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    x=Variable(datax)
    y= 5
# =============================================================================
#   define custom block
# =============================================================================
    power_block=power()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=x**y

# =============================================================================
#   define expected output power & product rule: (x^y)' = yx'x^(y-1)
# =============================================================================
    data_true=np.power(datax,y)
    gradient_true = y*x.gradient*datax**(y-1)
# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong div data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong div gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)