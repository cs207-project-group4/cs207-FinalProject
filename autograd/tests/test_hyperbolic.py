# -*- coding: utf-8 -*-
from autograd.blocks.hyperbolic import sinh
from autograd.variable import Variable
import numpy as np

def test_sinh():
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    sinh_block=sinh()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=sinh_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.sinh(data)
    gradient_true=np.diag(np.cosh(data))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)