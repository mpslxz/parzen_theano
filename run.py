from utils.windows import hypercube_window_func as HCW
from utils.kernels import hypercube as HCK
from core.parzen import ParzenWindow
from utils.viz import ParzenViz


if __name__ == "__main__":
    import numpy as np
    P = ParzenWindow(kernel_fcn=HCK, window_fcn=HCW, window_size=2.)
    x_train_normal = np.random.multivariate_normal(np.zeros(2),
                                                   np.eye(2),
                                                   1000).astype('float32')
    x_sample = np.array([[0.5, 0]]).astype('float32')

    print 30 * "-"
    print "Exp. 1"
    print "x_train \t--> 1000 2D points from multivariate N([0, 0], [[1,0],[0,1]])"
    print "x \t\t--> ", x_sample
    print "p(x) \t\t--> ", P.estimate_probability(x_sample, x_train_normal)
    print 30 * "-"

    x_train_uniform = np.random.uniform(-3, 3, (1000, 2)).astype('float32')
    x_sample = np.array([[0.5, 0]]).astype('float32')
    print "Exp. 2"
    print "x_train \t--> 1000 2D points from uniform U(-1, 1)"
    print "x \t\t--> ", x_sample
    print "p(x) \t\t--> ", P.estimate_probability(x_sample, x_train_uniform)
    print 30 * "-"

    viz_engine = ParzenViz(
        dist_list=[x_train_normal, x_train_uniform],
        dist_names=['normal', 'uniform'],
        parzen_obj=P)
    viz_engine.gen_plots()
