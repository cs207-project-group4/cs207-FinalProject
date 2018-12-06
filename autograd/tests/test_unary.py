from autograd.blocks.trigo import sin
from autograd.variable import Variable
import numpy as np

def test_neg():
    # =============================================================================
    #   define the input variable
    # =============================================================================
    data = np.random.random(5)
    x = Variable(data)
    
    y = Variable(-data, -x.gradient)

    # =============================================================================
    #   assert data pass
    # =============================================================================
    assert np.equal(-x.data, y.data).all(),'Data failed'

    # =============================================================================
    #   assert gradient forward pass
    # =============================================================================
    assert np.equal(-x.gradient,y.gradient).all(),'Gradient failed'
