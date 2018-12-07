# -*- coding: utf-8 -*-

from autograd.blocks.trigo import sin
from autograd.blocks.operations import add
from autograd.blocks.operations import sum_elts
from autograd.variable import Variable
import numpy as np
import autograd as ad





#ad.mode = 'reverse'
#x=Variable([1,2,3])
#block=sin()
#y=block(x)
#
#ab=add()
#
#z=ab(x,y)
#
#u=z+1
#
#sm=sum_elts()
#g=sm(u)
#
#g.backward()















def test_sin_reverse():
    #ad.mode='reverse'
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


