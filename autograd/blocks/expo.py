from autograd.blocks.block import SimpleBlock
import numpy as np


class exp(SimpleBlock):
    """
    vectorized sinus function on vectors
    """

    def data_fn(self, args):
        new_data = np.exp(args.data)
        return (new_data)

    def gradient_fn(self, args):
        grad = np.exp(args.data)
        return (grad)

class log(SimpleBlock):
    """
    vectorized sinus function on vectors
    """

    def data_fn(self, args):
        new_data = np.log(args.data)
        return (new_data)

    def gradient_fn(self, args):
        grad = 1/args.data
        return (grad)
