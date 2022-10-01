import numpy as np
import matplotlib.pyplot as plt

import icesat_lake_classification.utils as utl

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