# -*- coding: utf-8 -*-
from autograd.blocks.trigo import sin
from autograd.blocks.trigo import cos
from autograd.blocks.trigo import tan
from autograd.blocks.trigo import arcsin
from autograd.blocks.trigo import arccos
from autograd.blocks.trigo import arctan
from autograd.variable import Variable
import numpy as np
import autograd as ad

def test_sin_forward():
    ad.set_mode('forward')
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



def test_sin_reverse():
    ad.set_mode('reverse')
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
#   Compute gradient backwards
# =============================================================================
    y_block.backward()
    
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

   


def test_cos_forward():
    ad.set_mode('forward')
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



def test_cos_reverse():
    ad.set_mode('reverse')
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
    y_block.backward()
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



def test_tan_forward():
    ad.set_mode('forward')

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


def test_tan_reverse():
    ad.set_mode('reverse')

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
    y_block.backward()
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


def test_arcsin_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    arcsin_block=arcsin()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=arcsin_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.arcsin(data)
    gradient_true= np.diag(1/(np.sqrt(1 - data**2)))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong arcsin data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong arcsin gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)




def test_arcsin_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    arcsin_block=arcsin()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=arcsin_block(x)
    y_block.backward()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.arcsin(data)
    gradient_true= np.diag(1/(np.sqrt(1 - data**2)))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong arcsin data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong arcsin gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)





def test_arccos_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    arccos_block=arccos()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=arccos_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.arccos(data)
    gradient_true= np.diag(-1/(np.sqrt(1 - data**2)))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong arccos data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong arccos gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)



def test_arccos_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    arccos_block=arccos()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=arccos_block(x) 
    y_block.backward()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.arccos(data)
    gradient_true= np.diag(-1/(np.sqrt(1 - data**2)))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong arccos data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong arccos gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)




def test_arctan_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    arctan_block=arctan()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=arctan_block(x)

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.arctan(data)
    gradient_true= np.diag(1/(1 + data**2))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong arctan data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong arctan gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)


def test_arctan_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    arctan_block=arctan()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=arctan_block(x)
    y_block.backward()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.arctan(data)
    gradient_true= np.diag(1/(1 + data**2))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong arctan data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong arctan gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)
