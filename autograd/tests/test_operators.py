from autograd.blocks.trigo import sin
from autograd.variable import Variable
import numpy as np

def test_eq():
    # =============================================================================
    #   define the input variable
    # =============================================================================
    data = np.random.random(5)
    x = Variable(data)
    y = x

    # =============================================================================
    #   assert data pass
    # =============================================================================
    assert np.equal(x.data, y.data).all(),'Data failed'

    # =============================================================================
    #   assert gradient forward pass
    # =============================================================================
    assert np.equal(x.gradient,y.gradient).all(),'Gradient failed'

def test_neq():
    # =============================================================================
    #   define the input variable
    # =============================================================================
    data = np.random.random(5)
    x = Variable(data)
    y = Variable(data+1)

    # =============================================================================
    #   assert data pass
    # =============================================================================
    assert np.not_equal(x.data, y.data).all(),'Data failed'

def test_lt():
    # =============================================================================
    #   define the input variable
    # =============================================================================
    data = np.random.random(5)
    x = Variable(data)
    y = Variable(data+1)
    y.gradient = x.gradient + 1
    # =============================================================================
    #   assert data pass
    # =============================================================================
    assert np.less(x.data, y.data).all(),'Data failed'

    # =============================================================================
    #   assert gradient forward pass
    # =============================================================================
    assert np.less(x.gradient,y.gradient).all(),'Gradient failed'

def test_le():
    # =============================================================================
    #   define the input variable
    # =============================================================================
    data = np.random.random(5)
    x = Variable(data)
    y = Variable(data+1)
    y.gradient = x.gradient + 1
    # =============================================================================
    #   assert data pass
    # =============================================================================
    assert np.less_equal(x.data, y.data).all(),'Data failed'

    # =============================================================================
    #   assert gradient forward pass
    # =============================================================================
    assert np.less_equal(x.gradient,y.gradient).all(),'Gradient failed'

def test_gt():
    # =============================================================================
    #   define the input variable
    # =============================================================================
    data = np.random.random(5)
    x = Variable(data+1)
    y = Variable(data)
    x.gradient = y.gradient + 1
    # =============================================================================
    #   assert data pass
    # =============================================================================
    assert np.greater(x.data, y.data).all(),'Data failed'

    # =============================================================================
    #   assert gradient forward pass
    # =============================================================================
    assert np.greater(x.gradient,y.gradient).all(),'Gradient failed'

def test_ge():
    # =============================================================================
    #   define the input variable
    # =============================================================================
    data = np.random.random(5)
    x = Variable(data+1)
    y = Variable(data)
    x.gradient = y.gradient + 1
    # =============================================================================
    #   assert data pass
    # =============================================================================
    assert np.greater_equal(x.data, y.data).all(),'Data failed'

    # =============================================================================
    #   assert gradient forward pass
    # =============================================================================
    assert np.greater_equal(x.gradient,y.gradient).all(),'Gradient failed'
