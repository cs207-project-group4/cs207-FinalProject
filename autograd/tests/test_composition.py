from autograd.blocks.trigo import sin
from autograd.blocks.expo import exp
from autograd.variable import Variable
import numpy as np
import autograd as ad

def test_comp_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    sin_block = sin()
    exp_block = exp()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=sin_block(exp_block(x))
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.sin(np.exp(data))
    gradient_true=np.exp(data)*np.cos(np.exp(data))*np.identity(5)

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)




def test_comp_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    sin_block = sin()
    exp_block = exp()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=sin_block(exp_block(x))
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.sin(np.exp(data))
    gradient_true=np.exp(data)*np.cos(np.exp(data))*np.identity(5)

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)
