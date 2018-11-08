from  autograd.blocks.block import SimpleBlock
import numpy as np

class sinh(SimpleBlock):
    """
    vectorized sin h function on vectors
    """
    def data_fn(self, args):
        new_data = np.sinh(args.data)
        return(new_data)

    def gradient_fn(self, args):
        grad = np.cosh(args.data)
        return(grad)

class cosh(SimpleBlock):
    """
    vectorized cosine h function on vectors
    """
    def data_fn(self, args):
        new_data = np.cosh(args.data)
        return(new_data)

    def gradient_fn(self, args):
        grad = np.sinh(args.data)
        return(grad)

class tanh(SimpleBlock):
    """
    vectorized tan h function on vectors
    """
    def data_fn(self, args):
        new_data = np.tanh(args.data)
        return(new_data)

    def gradient_fn(self, args):
        grad = 1 - np.tanh(args.data)**2
        return(grad)