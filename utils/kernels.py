import theano.tensor as T


def hypercube(h, x, x_i):
    """hypercube kernel for the parzen window

    :param h: size of the hypercube
    :param x: training samples (n x d )
    :param x_i: point for density estimation (1 x d)
    :returns: elemwise distance between point to all of the samples (n x d)

    """
    x_i = T.addbroadcast(x_i, 0)
    return (x_i - x) / h
