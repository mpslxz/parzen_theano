import numpy as np
import matplotlib.pyplot as plt


class ParzenViz(object):

    def __init__(self, dist_list, dist_names, parzen_obj, size=5):
        self.dist_list = dist_list
        self.dist_names = dist_names
        self.parzen = parzen_obj
        self.size = size

    def gen_plots(self):
        frac = 0.2
        p = np.arange(-self.size, self.size, frac)
        map_size = (len(self.dist_list), 2 * int(
            self.size / frac), 2 * int(self.size / frac))
        dist = np.zeros(map_size)
        for d in range(len(self.dist_list)):
            for i in range(map_size[1]):
                for j in range(map_size[2]):
                    dist[d][i, j] = self.parzen.estimate_probability(
                        np.array([[p[i], p[j]]]).astype('float32'), self.dist_list[d])
        plt_count = 1
        for d in range(len(self.dist_list)):
            plt.subplot(len(self.dist_list), 2, plt_count)
            plt.scatter(
                self.dist_list[d][:, 0], self.dist_list[d][:, 1], s=0.1)
            plt.xlim([-self.size, self.size])
            plt.ylim([-self.size, self.size])
            plt.title(self.dist_names[d])
            plt_count += 1
            plt.subplot(len(self.dist_list), 2, plt_count)
            cs = plt.contour(p, p, dist[d], cmap='jet')
            plt.clabel(cs, inline=1)
            plt.title('p(x) of ' + self.dist_names[d])
            plt_count += 1
        plt.tight_layout()
        plt.savefig('plot.png')
