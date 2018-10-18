# -*- coding: utf-8 -*-\
import numpy as np
from autograd.variable import Variable

class Block():
    
    def data_fn(self,*args):
        raise NotImplementedError
    
    def gradient_fn(self, *args):
        raise NotImplementedError
    
    def __call__(self, *args):
        new_data=self.data_fn(*args)
        new_grad=self.gradient_fn(*args)
        
        return(Variable(new_data, new_grad))


class sin(Block):
    
    def data_fn(self, *args):        
        new_data = np.sin(args[0].data)       
        return(new_data)
    
    
    def gradient_fn(self, *args):
        new_grad = np.dot( np.cos(args[0].data), args[0].gradient)
        return(new_grad)
        
class mul(Block):
    def data_fn(self, *args):   
        #ASSERT SHAPE FOR DATA AND GRAD
        new_data = np.dot(args[0].data, args[1].data)      
        return(new_data)
    
    
    def gradient_fn(self, *args):
        new_grad = np.add ( np.dot( args[0].gradient, args[1].data ) , np.dot( args[0].data, args[1].gradient ) )
        return(new_grad)

class add(Block):
    def data_fn(self, *args):   
        #ASSERT SHAPE FOR DATA AND GRAD
        new_data = np.add(args[0].data, args[1].data)      
        return(new_data)
    
    
    def gradient_fn(self, *args):
        new_grad = np.add ( args[0].gradient, args[1].gradient ) 
        return(new_grad)
    
        
        
        

if __name__== '__main__':
    
    sinBlock=sin()
    mulBlock = mul()
    addBlock = add()
    
    print('test sinus')
    x=Variable(3)
    y=sinBlock(x)    
    print(y)
    
    print('')
    
    print('test mulitpl')    
    z=mulBlock(x,y)
    print(z)
    
    print('')
    
    print('test add')    
    t=addBlock(z,y)
    print(t)
    