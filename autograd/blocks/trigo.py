from autograd.blocks.block import SimpleBlock
import numpy as np

class sin(SimpleBlock):
    """
    vectorized sinus function on vectors
    """

    def data_fn(self, args):
        new_data = np.sin(args.data)
        return(new_data)


    def gradient_fn(self, args):
        grad = np.cos(args.data)
        return(grad)


class cos(SimpleBlock):
    """
    vectorized cosine function on vectors
    """
    def data_fn(self, args):
        new_data = np.cos(args.data)
        return(new_data)

    def gradient_fn(self, args):
        grad = -np.sin(args.data)
        return(grad)

class tan(SimpleBlock):
    """
    vectorized cosine function on vectors
	"""
    def data_fn(self, args):
        new_data = np.tan(args.data)
        return(new_data)

    def gradient_fn(self, args):
        grad = 1/np.cos(args.data)**2
        return(grad)

class arcsin(SimpleBlock):
    """
    vectorized arcsin function on vectors
    """

    def data_fn(self, args):
        new_data = np.arcsin(args.data)
        return(new_data)


    def gradient_fn(self, args):
        grad = 1/(np.sqrt(1 - args.data**2))
        return(grad)

class arccos(SimpleBlock):
    """
    vectorized arcsin function on vectors
    """

    def data_fn(self, args):
        new_data = np.arccos(args.data)
        return(new_data)


    def gradient_fn(self, args):
        grad = -1/(np.sqrt(1 - args.data**2))
        return(grad)

class arctan(SimpleBlock):
    """
    vectorized arcsin function on vectors
    """

    def data_fn(self, args):
        new_data = np.arctan(args.data)
        return(new_data)


    def gradient_fn(self, args):
        grad = 1/(1 + args.data**2)
        return(grad)
