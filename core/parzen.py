import theano
import theano.tensor as T
import numpy as np


class ParzenWindow(object):

    def __init__(self, kernel_fcn, window_fcn, window_size=1.):
        self.window_size_val = window_size
        self.window_size = T.constant(window_size)
        x = T.matrix()
        x_i = T.matrix()
        self.kernel_fcn = kernel_fcn
        self.window_fcn = window_fcn

        self._sample_prob, _ = theano.scan(sequences=x_i,
                                           non_sequences=x,
                                           fn=self._sample_probability)
        self._get_prob_op = theano.function([x_i, x], self._sample_prob)

    def _sample_probability(self, x_i, x):
        x_i = x_i.reshape((1, -1))
        x_i = T.addbroadcast(x_i, 0)
        x__ = self.kernel_fcn(h=self.window_size, x=x, x_i=x_i)
        k_n = self.window_fcn(x__, h=self.window_size).sum()
        return (1. * k_n / x.shape[0]) / (self.window_size ** x.shape[1])

    def estimate_probability(self, x_sample, x_train):
        return self._get_prob_op(x_sample, x_train)

    def max_posterior(self, x_train, y_train, x_sample):
        labels = np.unique(y_train)
        class_probs = np.zeros((x_sample.shape[0], len(labels)))
        for idx, l in enumerate(labels):
            class_probs[:, idx] = self.estimate_probability(
                x_sample, x_train[np.where(y_train == l)[0]])
        max_idx = np.argmax(class_probs, axis=1)
        return labels[max_idx], [i[max_idx[j]] for j, i in enumerate(class_probs)], class_probs
