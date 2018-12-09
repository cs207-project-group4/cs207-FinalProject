from autograd.blocks.trigo import sin
from autograd.variable import Variable
import numpy as np
import autograd as ad

def test_neg_forward():
    ad.set_mode('forward')
    # =============================================================================
    #   define the input variable
    # =============================================================================
    data = np.random.random(5)
    x = Variable(data)
    
    y_true = Variable(-x.data, -x.gradient)

    y_block=-x

    
    # =============================================================================
    #   assert data pass
    # =============================================================================
    assert np.equal(y_true.data, y_block.data).all(),'Data failed'

    # =============================================================================
    #   assert gradient forward pass
    # =============================================================================
    assert np.equal(y_true.gradient,y_block.gradient).all(),'Gradient failed'


def test_neg_reverse():
    ad.set_mode('reverse')
    # =============================================================================
    #   define the input variable
    # =============================================================================
    data = np.random.random(5)
    x = Variable(data)
    
    y_true = Variable(-x.data, np.diag([-1]*5))

    y_block=-x

    
    # =============================================================================
    #   assert data pass
    # =============================================================================
    assert np.equal(y_true.data, y_block.data).all(),'Data failed'

    # =============================================================================
    #   assert gradient forward pass
    # =============================================================================
    assert np.equal(y_true.gradient,y_block.gradient).all(),'Gradient failed'
