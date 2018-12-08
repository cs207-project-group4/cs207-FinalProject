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
    def __init__(self, base=np.e):
        super().__init__()
        self.base=base

    def data_fn(self, args):
        if base == np.e:
            new_data = np.log(args.data)
        elif base == 10:
            new_data = np.log10(args.data)
        elif base == 2:
            new_data = np.log2(args.data)
        else:
            raise ValueError('Encountered unsupported base value in `log`')

        return (new_data)

    def gradient_fn(self, args):
        grad = 1/args.data
        return (grad)
