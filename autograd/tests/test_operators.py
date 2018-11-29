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

