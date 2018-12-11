# -*- coding: utf-8 -*-
import autograd as ad
from autograd.variable import Variable
import numpy as np
from autograd.blocks.trigo import sin
from autograd.blocks.expo import exp
from autograd import config

def test_multiple_forward():
    """
    assert that the package works well when we use it repetively
    """
    
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



# =============================================================================
#     assert multiple gradient computes work
# =============================================================================
    for _ in range(5):
        y_block.compute_gradients()
        assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)
        assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)



# =============================================================================
#     assert multiple passes work
# =============================================================================
    for _ in range(5):
        y_block=sin_block(exp_block(x))
        y_block.compute_gradients()
        assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)
        assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

# =============================================================================
#   assert multiple definitions work
# =============================================================================
    for _ in range(5):
        
        data=np.random.random(5)
        x=Variable(data)

        sin_block = sin()
        exp_block = exp()
    
        y_block=sin_block(exp_block(x))
        y_block.compute_gradients()
  
    
        data_true=np.sin(np.exp(data))
        gradient_true=np.exp(data)*np.cos(np.exp(data))*np.identity(5)
    
  
        assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)
        assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)



def test_multiple_reverse():
    """
    assert that the package works well when we use it repetively
    """
    
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



# =============================================================================
#     assert multiple gradient computes work
# =============================================================================
    for _ in range(5):
        ad.reset_graph()
        y_block.compute_gradients()
        assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)
        assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)



# =============================================================================
#     assert multiple passes work
# =============================================================================
    for _ in range(5):
        
        ad.reset_graph()
        y_block=sin_block(exp_block(x))
        y_block.compute_gradients()
        assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)
        assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

# =============================================================================
#   assert multiple definitions work
# =============================================================================
    for _ in range(5):
        
        ad.reset_graph()
        data=np.random.random(5)
        x=Variable(data)

        sin_block = sin()
        exp_block = exp()
    
        y_block=sin_block(exp_block(x))
        y_block.compute_gradients()
  
    
        data_true=np.sin(np.exp(data))
        gradient_true=np.exp(data)*np.cos(np.exp(data))*np.identity(5)
    
  
        assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)
        assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)

