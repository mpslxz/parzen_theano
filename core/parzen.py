import theano
import theano.tensor as T


class ParzenWindow(object):

    def __init__(self, kernel_fcn, window_fcn, window_size=1.):
        self.window_size_val = window_size
        self.window_size = T.constant(window_size)
        x = T.matrix()
        x_i = T.matrix()

        x__ = kernel_fcn(h=self.window_size, x=x, x_i=x_i)
        k_n = window_fcn(x__, h=self.window_size).sum()
        self.get_win_samples = theano.function([x, x_i], k_n)

    def estimate_probability(self, x_train, x_sample):
        k_n = self.get_win_samples(x_train, x_sample)
        return (1. * k_n / len(x_train)) / (self.window_size_val ** x_train.shape[1])
