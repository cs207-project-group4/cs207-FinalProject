from autograd.blocks.operations import add
from autograd.blocks.operations import subtract
from autograd.blocks.operations import multiply
from autograd.blocks.operations import divide
from autograd.blocks.operations import power
from autograd.blocks.operations import sum_elts

from autograd.variable import Variable
import numpy as np
import autograd as ad

def test_add_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    
    
    
    x,y =Variable.multi_variables(datax, datay)
# =============================================================================
#   define custom block
# =============================================================================
    add_block=add()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=add_block(x,y)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.add(datax,datay)
    
    
    gradient_true_x=np.identity(5)
    gradient_true_y=np.identity(5)
    
# =============================================================================
#   assert forward pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_block.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_block.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])



# =============================================================================
#   assert overloading
# =============================================================================
    y_overloaded = x+y
    y_overloaded.compute_gradients()
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    y_overloaded = x+1
    y_overloaded.compute_gradients()
    
    data_true=np.add(datax,1)   
    gradient_true_x=np.identity(5)
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    y_overloaded = 1+x
    y_overloaded.compute_gradients()
    
    data_true=np.add(datax,1)   
    gradient_true_x=np.identity(5)
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass for x . expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass for y . expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])






def test_add_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    
    
    
    x,y =Variable.multi_variables(datax, datay)
# =============================================================================
#   define custom block
# =============================================================================
    add_block=add()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=add_block(x,y)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.add(datax,datay)
    
    
    gradient_true_x=np.identity(5)
    gradient_true_y=np.identity(5)
    
# =============================================================================
#   assert forward pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_block.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_block.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])



# =============================================================================
#   assert overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = x+y
    y_overloaded.compute_gradients()
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = x+1
    y_overloaded.compute_gradients()
    
    data_true=np.add(datax,1)   
    gradient_true_x=np.identity(5)
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass for x . expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass for y . expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])



# =============================================================================
#   assert scalar overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = 1+x
    y_overloaded.compute_gradients()
    
    data_true=np.add(datax,1)   
    gradient_true_x=np.identity(5)
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass for x . expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass for y . expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])





def test_subtract_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    
    
    
    x,y =Variable.multi_variables(datax, datay)
# =============================================================================
#   define custom block
# =============================================================================
    sub_block=subtract()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=sub_block(x,y)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.subtract(datax,datay)
    
    
    gradient_true_x=np.identity(5)
    gradient_true_y=-np.identity(5)
    
# =============================================================================
#   assert forward pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_block.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_block.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])



# =============================================================================
#   assert overloading
# =============================================================================
    y_overloaded = x-y
    y_overloaded.compute_gradients()
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    y_overloaded = x-1
    y_overloaded.compute_gradients()
    
    data_true=np.subtract(datax,1)   
    gradient_true_x=np.identity(5)
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    y_overloaded = 1-x
    y_overloaded.compute_gradients()
    
    data_true=np.subtract(1, datax)   
    gradient_true_x=-np.identity(5)
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass for x . expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass for y . expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])




def test_subtract_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    
    
    
    x,y =Variable.multi_variables(datax, datay)
# =============================================================================
#   define custom block
# =============================================================================
    sub_block=subtract()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=sub_block(x,y)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.subtract(datax,datay)   
    gradient_true_x=np.identity(5)
    gradient_true_y=-np.identity(5)
    
# =============================================================================
#   assert forward pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_block.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_block.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])



# =============================================================================
#   assert overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = x-y
    y_overloaded.compute_gradients()
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = x-1
    y_overloaded.compute_gradients()
    
    data_true=np.subtract(datax,1)   
    gradient_true_x=np.identity(5)
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass for x . expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass for y . expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])



# =============================================================================
#   assert scalar overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = 1-x
    y_overloaded.compute_gradients()
    
    data_true=np.subtract(1, datax)   
    gradient_true_x=-np.identity(5)
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass for x . expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass for y . expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])


def test_multiply_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    
    
    
    x,y =Variable.multi_variables(datax, datay)
# =============================================================================
#   define custom block
# =============================================================================
    mul_block=multiply()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=mul_block(x,y)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.multiply(datax,datay)
    
    
    gradient_true_x=np.diag(y.data)
    gradient_true_y=np.diag(x.data)
    
# =============================================================================
#   assert forward pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_block.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_block.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])



# =============================================================================
#   assert overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = x*y
    y_overloaded.compute_gradients()
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = x*4
    y_overloaded.compute_gradients()
    
    data_true=np.multiply(datax,4)   
    gradient_true_x=np.identity(5)*4
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = 4*x
    y_overloaded.compute_gradients()
    
    data_true=np.multiply(4, datax)   
    gradient_true_x=4*np.identity(5)
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass for x . expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass for y . expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])



def test_multiply_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    
    
    
    x,y =Variable.multi_variables(datax, datay)
# =============================================================================
#   define custom block
# =============================================================================
    mul_block=multiply()

# =============================================================================
#   compute output of custom block
# =============================================================================
    print(ad.mode)
    y_block=mul_block(x,y)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.multiply(datax,datay)
    
    
    gradient_true_x=np.diag(y.data)
    gradient_true_y=np.diag(x.data)
    
# =============================================================================
#   assert forward pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_block.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_block.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])



# =============================================================================
#   assert overloading
# =============================================================================
    y_overloaded = x*y
    y_overloaded.compute_gradients()
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    y_overloaded = x*4
    y_overloaded.compute_gradients()
    
    data_true=np.multiply(datax,4)   
    gradient_true_x=np.identity(5)*4
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    y_overloaded = 4*x
    y_overloaded.compute_gradients()
    
    data_true=np.multiply(4, datax)   
    gradient_true_x=4*np.identity(5)
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass for x . expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass for y . expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])



def test_divide_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    
    
    
    x,y =Variable.multi_variables(datax, datay)
# =============================================================================
#   define custom block
# =============================================================================
    div_block=divide()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=div_block(x,y)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.divide(datax,datay)
    
    
    gradient_true_x=np.diag(1/y.data)
    gradient_true_y=-np.diag(x.data/(y.data**2))
    
# =============================================================================
#   assert forward pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.allclose(gradient_true_x, y_block.gradient[0]), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.allclose(gradient_true_y, y_block.gradient[1]), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])



# =============================================================================
#   assert overloading
# =============================================================================
    y_overloaded = x/y
    y_overloaded.compute_gradients()
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.allclose(gradient_true_x, y_overloaded.gradient[0]), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.allclose(gradient_true_y, y_overloaded.gradient[1]), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    y_overloaded = x/4
    y_overloaded.compute_gradients()
    
    data_true=np.divide(datax,4)   
    gradient_true_x=np.identity(5)/4
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    y_overloaded = 4/x
    y_overloaded.compute_gradients()
    
    data_true=np.divide(4, datax)   
    gradient_true_x=-4*np.diag(1/(x.data**2))
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.allclose(gradient_true_x, y_overloaded.gradient[0]), 'wrong add gradient forward pass for x . expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass for y . expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])


def test_divide_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    datay=np.random.random(5)
    
    
    
    x,y =Variable.multi_variables(datax, datay)
# =============================================================================
#   define custom block
# =============================================================================
    div_block=divide()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=div_block(x,y)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.divide(datax,datay)
    
    
    gradient_true_x=np.diag(1/y.data)
    gradient_true_y=-np.diag(x.data/(y.data**2))
    
# =============================================================================
#   assert forward pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.allclose(gradient_true_x, y_block.gradient[0]), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.allclose(gradient_true_y, y_block.gradient[1]), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])



# =============================================================================
#   assert overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = x/y
    y_overloaded.compute_gradients()
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.allclose(gradient_true_x, y_overloaded.gradient[0]), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.allclose(gradient_true_y, y_overloaded.gradient[1]), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = x/4
    y_overloaded.compute_gradients()
    
    data_true=np.divide(datax,4)   
    gradient_true_x=np.identity(5)/4
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true_x, y_overloaded.gradient[0]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_x,y_block.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true_y,y_block.gradient[1])


# =============================================================================
#   assert scalar overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = 4/x
    y_overloaded.compute_gradients()
    
    data_true=np.divide(4, datax)   
    gradient_true_x=-4*np.diag(1/(x.data**2))
    gradient_true_y=np.zeros((5,5))
    
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.allclose(gradient_true_x, y_overloaded.gradient[0]), 'wrong add gradient forward pass for x . expected {}, given{}'.format(gradient_true_x,y_overloaded.gradient[0])
    assert np.equal(gradient_true_y, y_overloaded.gradient[1]).all(), 'wrong add gradient forward pass for y . expected {}, given{}'.format(gradient_true_y,y_overloaded.gradient[1])





def test_power_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    x =Variable(datax)
    n=4
    
    
# =============================================================================
#   define custom block
# =============================================================================
    power_block=power()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=power_block(x,power_exponent=n)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.power(datax,n)
    
    
    gradient_true=np.diag(n*x.data**(n-1))
    
# =============================================================================
#   assert forward pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)



# =============================================================================
#   assert overloading
# =============================================================================
    y_overloaded = x**n
    y_overloaded.compute_gradients()
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true, y_overloaded.gradient).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)



def test_power_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variable
# =============================================================================
    datax=np.random.random(5)
    x =Variable(datax)
    n=4
    
    
# =============================================================================
#   define custom block
# =============================================================================
    power_block=power()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=power_block(x,power_exponent=n)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.power(datax,n)
    
    
    gradient_true=np.diag(n*x.data**(n-1))
    
# =============================================================================
#   assert forward pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)



# =============================================================================
#   assert overloading
# =============================================================================
    ad.reset_graph()
    y_overloaded = x**n
    y_overloaded.compute_gradients()
    assert np.equal(data_true, y_overloaded.data).all(), 'wrong add data pass. expected {}, given{}'.format(data_true, y_block.data)
    assert np.equal(gradient_true, y_overloaded.gradient).all(), 'wrong add gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)




def test_sum_elts_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variablet
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    sum_block=sum_elts()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=sum_block(x)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.sum(data)
    gradient_true=np.ones((1,5))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sinh data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sinh gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)


def test_sum_elts_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variablet
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   define custom block
# =============================================================================
    sum_block=sum_elts()

# =============================================================================
#   compute output of custom block
# =============================================================================
    y_block=sum_block(x)
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=np.sum(data)
    gradient_true=np.ones((1,5))

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sinh data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sinh gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)





def test_extract_intger_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variablet
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   compute output of operation
# =============================================================================
    y_block=x[2]
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=data[2]
    gradient_true=np.array([0,0,1,0,0])

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sinh data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sinh gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)





def test_extract_intger_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variablet
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   compute output of operation
# =============================================================================
    y_block=x[2]
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=data[2]
    gradient_true=np.array([0,0,1,0,0])

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sinh data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sinh gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)




def test_extract_slice_forward():
    ad.set_mode('forward')
# =============================================================================
#   define the input variablet
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   compute output of operation
# =============================================================================
    y_block=x[1:3]
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=data[1:3]
    gradient_true=np.array([[0,1,0,0,0],
                            [0,0,1,0,0]])

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sinh data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sinh gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)



def test_extract_slice_reverse():
    ad.set_mode('reverse')
# =============================================================================
#   define the input variablet
# =============================================================================
    data=np.random.random(5)
    x=Variable(data)

# =============================================================================
#   compute output of operation
# =============================================================================
    y_block=x[1:3]
    y_block.compute_gradients()
# =============================================================================
#   define expected output
# =============================================================================
    data_true=data[1:3]
    gradient_true=np.array([[0,1,0,0,0],
                            [0,0,1,0,0]])

# =============================================================================
#   assert data pass
# =============================================================================
    assert np.equal(data_true, y_block.data).all(), 'wrong sinh data pass. expected {}, given{}'.format(data_true, y_block.data)

# =============================================================================
#   assert gradient forward pass
# =============================================================================
    assert np.equal(gradient_true, y_block.gradient).all(), 'wrong sinh gradient forward pass. expected {}, given{}'.format(gradient_true,y_block.gradient)
