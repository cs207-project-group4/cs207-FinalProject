# -*- coding: utf-8 -*-
from  autograd.blocks.block import SimpleBlock
import numpy as np

class sin(SimpleBlock):
    """
    vectorized sinus function on vectors
    """
    
    def data_fn(self, args):        
        new_data = np.sin(args.data)       
        return(new_data)
    
    
    def gradient_fn(self, args):
        grad =np.cos(args.data)
        return(grad)
        
    
        

        