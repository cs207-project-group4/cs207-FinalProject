from autograd.blocks.expo import exp, logistic
from autograd.blocks.expo import log
from autograd.blocks.expo import sqrt
from autograd.variable import Variable
import autograd as ad
import numpy as np

def test_exp_forward():
    ad.set_mode('forward')
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
    y_block.compute_gradients()
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


def test_exp_reverse():
    ad.set_mode('reverse')
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
    y_block.compute_gradients()

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
    ad.set_mode('forward')


def test_log_forward():
    ad.set_mode('forward')
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
    y_block.compute_gradients()
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



def test_log_reverse():
    ad.set_mode('reverse')
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
    y_block.compute_gradients()

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
    ad.set_mode('forward')




def test_log_diffBase_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    base = np.random.randint(2,5) + np.random.random()
    log_block=log(base=base)

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=log_block(x)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.log(data)/np.log(base)
    gradient_true=np.diag(1/(data*np.log(base)))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong log data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong log gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)



def test_log_diffBase_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    base = np.random.randint(2,5) + np.random.random()
    log_block=log(base=base)

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=log_block(x)
    y_block.compute_gradients()

# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.log(data)/np.log(base)
    gradient_true=np.diag(1/(data*np.log(base)))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong log data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong log gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)
    ad.set_mode('forward')


def test_logistic_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    logistic_block=logistic()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=logistic_block(x)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=1/(1+np.exp(-data))
    gradient_true=np.diag(np.exp(data)/(1+np.exp(data))**2)

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)


def test_logistic_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    logistic_block=logistic()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=logistic_block(x)
    y_block.compute_gradients()

# =============================================================================
#   define expected output
# =============================================================================
    data_true=1/(1+np.exp(-data))
    gradient_true=np.diag(np.exp(data)/(1+np.exp(data))**2)

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong exp data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong exp gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)
    ad.set_mode('forward')