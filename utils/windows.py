import theano.tensor as T


def hypercube_window_func(x, h):
    """Checks if all of the elements in x fall inside the hypercube

    :param x: input matrix (n x d)
    :param h: size of the hypercube
    :returns: 1 if x_i is inside hypyecube, otherwise 0

    """

    measures = 1. * (abs(x) < (h / 2.))
    return T.int_div(T.sum(measures, axis=1),  measures.shape[1])
