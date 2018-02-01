import matplotlib.pyplot as plt
from utils.windows import hypercube_window_func as HCW
from utils.kernels import hypercube as HCK
from core.parzen import ParzenWindow

if __name__ == "__main__":
    import numpy as np
    P = ParzenWindow(kernel_fcn=HCK, window_fcn=HCW, window_size=1.)
    x_train = np.random.normal(0, 1., (10000, 2)).astype('float32')
    x_sample = np.array([[0, 0]]).astype('float32')

    print 30 * "-"
    print "x_train \t--> 10000 2D points from N(0, 1)"
    print "x \t\t--> ", x_sample.squeeze()
    print "p(x) \t\t--> ", P.estimate_probability(x_train, x_sample)
    print 30 * "-"

    size = 5
    frac = 0.2
    p = np.arange(-size, size, frac)
    map_size = (2 * int(size / frac), 2 * int(size / frac))
    dist = np.zeros(map_size)
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            dist[i, j] = P.estimate_probability(x_train,
                                                np.array([[p[i], p[j]]]).astype('float32'))
    plt.subplot(1, 2, 1)
    plt.scatter(x_train[:, 0], x_train[:, 1], s=0.01)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.title('points')
    plt.subplot(1, 2, 2)
    cs = plt.contour(p, p, dist, cmap='jet')
    plt.clabel(cs, inline=1)
    plt.title('p(x)')
    plt.savefig('plot.png')
