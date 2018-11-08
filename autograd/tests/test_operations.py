from autograd.blocks.operations import add
from autograd.blocks.operations import subtract
from autograd.blocks.operations import multiply
from autograd.blocks.operations import divide
from autograd.blocks.operations import power
from autograd.blocks.operations import dot

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
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)
