# -*- coding: utf-8 -*-
from autograd.blocks.trigo import sin
from autograd.blocks.trigo import cos
from autograd.blocks.trigo import tan
from autograd.variable import Variable
import numpy as np
import autograd as ad

def test_sin_forward():
    ad.mode='forward'
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    sin_block=sin()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=sin_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.sin(data)
    gradient_true=np.diag(np.cos(data))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sin data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sin gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)







def test_cos():
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    cos_block=cos()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=cos_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.cos(data)
    gradient_true=np.diag(-np.sin(data))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong cos data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong cos gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

def test_tan():
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    tan_block=tan()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=tan_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.tan(data)
    gradient_true=np.diag(1/np.cos(data)**2)

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong tan data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong tan gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)
