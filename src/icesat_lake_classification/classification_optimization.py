import numpy as np
import matplotlib.pyplot as plt

import icesat_lake_classification.utils as utl

class gaussian_kde(object):
    """
    Class which generates a gaussian kernel density for a 2d point cloud, functions extract statistics of the kde
    """

    def __init__(self,
                 x, y, grid_size=128
                 ):
        """
        Parameters
        ----------
        x : nd.array
            numpy array with the x dimension data for the KDE
        y : nd.array
            numpy array with the y dimension data for the KDE
        """

        import scipy.stats

        xy = np.vstack((x,y))
        dens = scipy.stats.gaussian_kde(xy)
        gx, gy = np.mgrid[x.min():x.max():128j, y.min():y.max():128j]
        gxy = np.dstack((gx, gy))  # shape is (128, 128, 2)
        z = np.apply_along_axis(dens, 2, gxy)
        z = z.reshape(128,128)

        self.z = z
        self.x = x
        self.y = y
        self.gx = gx
        self.gy = gy
        self.extent = [x.min(), x.max(), y.min(), y.max()]


    def mean(self):
        return np.mean(self.z)

    def min(self):
        return np.min(self.z)

    def max(self):
        return np.max(self.z)

    def median(self):
        return np.median(self.z)

    def plot_2d(self, outpath=None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.z.T, cmap=plt.cm.gist_earth_r, extent=self.extent, aspect='auto')

        ax.plot(self.x, self.y, 'k.', markersize=2)

        ax.set_xlim(self.extent[0:2])
        ax.set_ylim(self.extent[2:4])
        fig.colorbar(im)

        if outpath:
            plt.savefig(outpath)


    def plot_histogram(self, bins=100, outpath=None):
        fig, axs = plt.subplots(1, 1,
                                figsize=(10, 7))

        counts, bin_edges, _ = axs.hist(self.z.flatten(), bins=bins, cumulative=-1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        axs.plot(bin_centers, counts)
        import kneed
        kl = kneed.KneeLocator(bin_centers, counts, curve="convex", direction='decreasing',online=True)

        if outpath:
            plt.savefig(outpath)

    def get_histogram_peak(self, bins=100):
        counts, bins = np.histogram(self.z.flatten(), bins=bins)
        return bins[np.argmax(counts)]

    def get_histogram_peak2(self, bins=100):
        counts, bins = np.histogram(self.z.flatten(), bins=bins)
        counts.pop()
        return bins[np.argmax(counts)]



def find_optimal_eps(data, min_pts=1, method='max', outpath=None, strict=None):

    distances = utl.get_distance_NN_min_pts(data,min_pts=min_pts)
    if method == 'max':
        distance_desc = np.array(sorted(distances[:, min_pts - 1], reverse=True))
        distance_max = distance_desc.copy()
    elif method == 'average':
        distance_desc = np.array(sorted(np.mean(distances, axis=1), reverse=True))
        distance_max = np.array(sorted(distances[:, min_pts - 1], reverse=True))
    elif method == 'total':
        distance_desc = np.array(sorted(np.sum(distances, axis=1), reverse=True))
        distance_max = np.array(sorted(distances[:, min_pts - 1], reverse=True))
    x_array = np.array(list(range(1, len(distance_desc) + 1)))

    normalized_x = utl.min_max_normalize_array(x_array)
    normalized_y = utl.min_max_normalize_array(distance_desc)

    if strict:
        normalized_x = normalized_x / strict

    coordinates = np.array([[x,y] for x,y in zip(normalized_x, normalized_y)])

    distance_elbow = utl.calc_euclidean_distance_arr(np.array([0,0]), coordinates)

    elbow_coord = np.array([coordinates[np.argmin(distance_elbow)][0], coordinates[np.argmin(distance_elbow)][1]])
    value_normalized = utl.reverse_min_max_normalize_array(elbow_coord[1], distance_desc)
    index = utl.find_nearest(distance_desc, value_normalized)
    eps = distance_max[index]

    # print(eps, value_normalized)

    if outpath:
        f1, ax = plt.subplots(figsize=(20, 20))
        ax.plot(normalized_x, normalized_y)
        ax.scatter(elbow_coord[0], elbow_coord[1])
        ax.set_xlabel('normalized number of photons')
        ax.set_ylabel('normalized distance to photon # {}'.format(min_pts))
        plt.savefig(outpath)

    return eps