from autograd.blocks.expo import exp
from autograd.blocks.expo import log
from autograd.blocks.expo import sqrt
from autograd.variable import Variable
import numpy as np

def test_exp():
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    exp_block=exp()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=exp_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.exp(data)
    gradient_true=np.diag(np.exp(data))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)


def test_log():
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    log_block=log()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=log_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.log(data)
    gradient_true=np.diag(1/(data))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong log data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong log gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

def test_exp():
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    sqrt_block=sqrt()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=sqrt_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.sqrt(data)
    gradient_true=np.diag(1/(2*np.sqrt(data)))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sqrt data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sqrt gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)
