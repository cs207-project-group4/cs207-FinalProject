# -*- coding: utf-8 -*-
import autograd as ad
import numpy as np

class Node():
    def __init__(self, output_dim=None):
        
        #ID of the node
        if ad.ids==[]:
            #first node we create
            self.id=0
            ad.ids+=[0]
            
        else:
            self.id=ad.ids[-1]+1
            ad.ids+=[self.id]
            
            
        #dimension of the current variable associated to this node
        self.output_dim=output_dim
        
        #number of times this node has been used
        self.times_used=0
        
        #number of times this node has been visited in the backprop
        self.times_visited=0
            
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


    def backward(self):
        """
        implement reverse AD, return the gradient of current variable w.r. input Variables
        input variables are the ones who don't have any childrens
        """
        #initiate the gradients
        print('node {} grad {}'.format(self.id, self.gradient))
        print('node {} times visited : {}/{}'.format(self.id, self.times_visited, self.times_used))

        if self.gradient is None:
            self.gradient=np.eye(self.output_dim)
            self.times_visited+=1

            
            
            if self.childrens==[]:
                return(self.gradient)
            else:
                return(self.backward())
            
        else:            
            if self.childrens!=[]:
                #we can still going deeper in backprop
                
                for child in self.childrens:
                    node,jacobian=child['node'], child['jacobian']
                    
                    new_grad = np.dot(self.gradient, jacobian)
                    
                    if node.gradient is None:
                        node.gradient = new_grad
                    else:                        
                        node.gradient += new_grad
                        
                    node.times_visited+=1
                        
                    if node.times_used ==node.times_visited:                       
                        return(node.backward())                
                    else:
                        pass
                    
            else:
                #terminal case, we return the gradients w.r. the input variable
                
                return(self.gradient)
            
            
            
        
        
        
        