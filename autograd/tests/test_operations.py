from autograd.blocks.operations import add
from autograd.blocks.operations import subtract
from autograd.blocks.operations import multiply
from autograd.blocks.operations import divide
from autograd.blocks.operations import power
from autograd.blocks.operations import sum_elts

from autograd.variable import Variable
import numpy as np

def test_add():
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
    add_block=add()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=add_block(x,y)
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.add(datax,datay)
    gradient_true=2*np.identity(5)

# =============================================================================
#   assert overloading
# =============================================================================
    y_overloaded = x+y
    assert np.equal(y_overloaded.data, y_block.data).all(), 'add overloading failed'
    assert np.equal(y_overloaded.gradient, y_block.gradient).all(), 'add overloading failed'

# =============================================================================
#   assert scalar overloading
# =============================================================================
    scalar = Variable(1)
    y_scale = add_block(scalar,x)
    y_scale_overloaded = 1+x
    assert np.equal(y_scale.data, y_scale_overloaded.data).all(), 'add scalar overloading failed'
    assert np.equal(y_overloaded.gradient, y_block.gradient).all(), 'add overloading failed'

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)




def test_subtract():
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
    y_block=subtract_block(x,y)
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.subtract(datax,datay)
    gradient_true=np.zeros((5,5))

# =============================================================================
#   assert overloading
# =============================================================================
    y_overloaded = x-y
    assert np.equal(y_overloaded.data, y_block.data).all(), 'sub overloading failed'
    assert np.equal(y_overloaded.gradient, y_block.gradient).all(), 'sub overloading failed'

# =============================================================================
#   assert scalar overloading
# =============================================================================
    scalar = Variable(1)

    y_scale = subtract_block(scalar,x)
    y_scale_overloaded = 1-x
    assert np.equal(y_scale.data, y_scale_overloaded.data).all(), 'sub scalar overloading failed'
    assert np.equal(y_overloaded.gradient, y_block.gradient).all(), 'sub overloading failed'


# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sub data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sub gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

def test_multiply():
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
    y_block=multiply_block(x,y)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.multiply(datax,datay)
    gradient_true=np.identity(5)*y.data + np.identity(5)*x.data

# =============================================================================
#   assert overloading
# =============================================================================
    y_overloaded = x*y
    assert np.equal(y_overloaded.data, y_block.data).all(), 'mul overloading failed'
    assert np.equal(y_overloaded.gradient, y_block.gradient).all(), 'mul overloading failed'

# =============================================================================
#   assert scalar overloading
# =============================================================================
    scalar = Variable(5)

    y_scale = multiply_block(scalar,x)
    y_scale_overloaded = 5*x
    assert np.equal(y_scale.data, y_scale_overloaded.data).all(), 'mult scalar overloading failed'
    assert np.equal(y_overloaded.gradient, y_block.gradient).all(), 'mult overloading failed'

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong mult data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong mult gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

def test_divide():
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
    y_block=divide_block(x, y)
# =============================================================================
#   define expected output
# =============================================================================
    data_true =np.divide(datax,datay)

    gradient_true = (x.gradient*y.data-x.data*y.gradient)/y.data**2

# =============================================================================
#   assert overloading
# =============================================================================
    y_overloaded = x/y
    assert np.equal(y_overloaded.data, y_block.data).all(), 'div overloading failed'
    assert np.equal(y_overloaded.gradient, y_block.gradient).all(), 'div overloading failed'

# =============================================================================
#   assert scalar overloading
# =============================================================================
    scalar = Variable(5)

    y_scale = divide_block(scalar,x)
    y_scale_overloaded = 5/x
    assert np.equal(y_scale.data, y_scale_overloaded.data).all(), 'div scalar overloading failed'
    assert np.equal(y_overloaded.gradient, y_block.gradient).all(), 'div overloading failed'

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong div data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass, use allclose because division create small differences.  Outputs look same though!
# =============================================================================
    assert np.allclose(gradient_true, y_block.gradient, rtol = .001), 'wrong div gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)


def test_power():
# =============================================================================
#   define the input variable
# =============================================================================
    datax = np.random.random(5)
    x = Variable(datax)
    y = int(4)

# =============================================================================
#   define custom block
# =============================================================================
    power_block = power()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=power_block(x,y)
# =============================================================================
#   define expected output power & product rule: (x^y)' = yx'x^(y-1)
# =============================================================================
    data_true=np.power(datax,y)
    gradient_true = y*x.gradient*datax**(y-1)

# =============================================================================
#   assert overloading
# =============================================================================
    y_overloaded = x**y
    assert np.equal(y_overloaded.data, y_block.data).all(), 'power overloading failed'
    assert np.equal(y_overloaded.gradient, y_block.gradient).all(), 'power overloading failed'

# =============================================================================
#   assert scalar overloading
# =============================================================================
    scalar = 5

    y_scale = power_block(x,scalar)
    y_scale_overloaded = x**5
    assert np.equal(y_scale.data, y_scale_overloaded.data).all(), 'power scalar overloading failed'
    assert np.equal(y_overloaded.gradient, y_block.gradient).all(), 'power overloading failed'

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong div data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong div gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)



def test_sum_elts():
# =============================================================================
#   define the input variablet
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    sum_block=sum_elts()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=sum_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.sum(data)
    gradient_true=np.ones((1,5))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sinh data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sinh gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

