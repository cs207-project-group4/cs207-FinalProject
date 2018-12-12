# -*- coding: utf-8 -*-
import autograd as ad
import numpy as np
from autograd import config

class C_graph():
    """
    aggregating class for the nodes in the computational graph
    """
    
    def __init__(self, nodes=[]):
        self.ids=[]
        self.input_node=[]
        self.output_node=None
        self.input_shapes=[]
        
    def reset_graph(self):    
        #print('start cleaning')
        #self.input_node=[]
        #self.input_shapes=[]
       
        for node in config.list_of_nodes:          
            #remove all info from these nodes
            
            node.gradient=None
            node.times_visited=0
            node.times_used=0
            #_variable_.node.childrens={}
            
            #if _variable_.node.childrens!=[]:
                #this is not the root node
            #    _variable_.node.id=
                    
    def define_path(self, node):
        """
        make a first backward pass without doing any computation
        It is just meant to check which variables are involved in the computation of the node given
        """ 
        if node.childrens!=[]:
           for child in node.childrens:
                node_child = child['node']
                node_child.times_used+=1
                self.define_path(node_child)
        
        
        #take care of not used nodes, set their gradient to 0
        for node in self.input_node:
            if node.times_used==0:
                node.gradient=np.zeros((node.output_dim, self.output_node.output_dim))
            
        
                
        
        
    
        
    
        
        
        

class Node():
    """
    basic element of the computational graph
    """
    def __init__(self, output_dim=None):
        
        #ID of the node
        if ad.c_graph.ids==[]:
            #first node we create
            self.id=0
            ad.c_graph.ids+=[0]
            
        else:
            self.id=ad.c_graph.ids[-1]+1
            ad.c_graph.ids+=[self.id]
            
            
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
        #print('')
        
        #print('node {} grad {}'.format(self.id, self.gradient))
        #print('node {} times visited : {}/{}'.format(self.id, self.times_visited, self.times_used))

        if self.gradient is None:
            self.gradient=np.eye(self.output_dim)
            self.times_visited+=1

            
            
            if self.childrens==[]:
                return(self.gradient)
            else:
                self.backward()
            
        else:            
            if self.childrens!=[]:
                #we can still going deeper in backprop
                #print(len(self.childrens), ' childrens', str([self.childrens[i]['node'].id for i in range(len(self.childrens))]))
                for child in self.childrens:
                    node,jacobian=child['node'], child['jacobian']
                    
                    new_grad = np.dot(self.gradient, jacobian)
                    #print(node.gradient)
                    #print(new_grad)
                    
                    if node.gradient is None:
                        node.gradient = new_grad
                    else:                        
                        node.gradient += new_grad
                        
                    node.times_visited+=1
                    #print('looking at node {} \ngradient {}'.format(node.id, node.gradient))

                        
                    if node.times_used ==node.times_visited:     
                        #print(node.gradient)
                        node.backward()          
                    else:
                        #still some computations to perform upwards before going deeped
                        #print('node {} visits : {}/{}'.format(node.id, node.times_visited, node.times_used))
                        pass
            
                
            
            
            
        
        
        
        