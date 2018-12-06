# -*- coding: utf-8 -*-
import autograd as ad

class Node():
    def __init__(self, ):
        
        #ID of the node
        if ad.ids==[]:
            #first node we create
            self.id=0
            ad.ids+=[0]
            
        else:
            self.id=ad.ids[-1]+1
            ad.ids+=[self.id]
            
        #store the gradients of the output node with respect to this node
        self.gradient=None
        
        #the childrens list stores the children node, as well as the jacobian used to go from child node
        # to this node
        #
        #for exemple, if we use the block multiply, then the output node of this block will have two children
        # y=x1*x2
        #
        # children 1 
        #-------------
        #   node : node associated to x1
        #   jacobian used to go from x1 to y : diag(x2) because y' = x1'*x2 + x1*x2'
        #
        #children 2
        #-----------
        #   node : node associated to x2
        #   jacobian used to go from x2 to y : diag(x1) because y' = x1'*x2 + x1*x2'
        #
        #
        # childrens = [{'node':Node, 'jacobian':Jacobian}, ...]
        
        self.childrens=[]
        
        
        
        