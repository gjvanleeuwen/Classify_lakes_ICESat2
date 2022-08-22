import logging

import numpy as np

from geopy.distance import geodesic


def set_log_level(log_level):

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.DEBUG
    logging.getLogger().setLevel(log_level)

def config_log(log_level='DEBUG', log_format='[%(process)d %(processName)s] %(asctime)s - %(levelname)s - %(message)s'):

    config_error = False
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.DEBUG
        config_error = True
    logging.basicConfig(format=log_format, level=numeric_level)
    if config_error:
        log('unable to config logging!!!', 'CRITICAL')

def log(msg, log_level='DEBUG'):

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.CRITICAL
    logging.log(numeric_level, msg)
    # print msg

def find_nearest(array, value):
    return np.abs(array - value).argmin()

def get_geodetic_distance(origin, destenation):
    # (latitude, longitude) don't confuse

    return geodesic(origin, destenation).meters

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def calc_euclidean_distance(origin, coord):
    return np.linalg.norm(origin - coord)

def calc_euclidean_distance_arr(origin, coordinates):
    return np.linalg.norm(origin - coordinates, axis=1)

def calc_euclidean_distance_matrix(coordinates):
    # return [euclidean_distance_array(origin, coordinates) for origin in coordinates]
    from scipy.spatial.distance import cdist
    return cdist(coordinates, coordinates)

def get_points_in_radius(points, origin, radius=1.0):
    import math

    x1, y1 = origin
    return [(x2,y2) for x2,y2 in points if math.sqrt((x1-x2)**2+(y1-y2)**2) <= radius]

def get_number_NN_radius(coords, radius=1.0):
    import scipy.spatial as spatial

    tree = spatial.KDTree(np.array(coords))

    neighbors = tree.query_ball_tree(tree, radius)
    frequency = np.array([len(i) for i in neighbors])
    return frequency

def get_density_NN_radius(coords, radius=1.0):
    return get_number_NN_radius(coords, radius) / radius**2

def get_distance_NN_min_pts(data, min_pts=1):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=min_pts).fit(data)
    distances, indices = nbrs.kneighbors(data)

    return distances

def get_average_distance_NN_min_pts(data, min_pts=1):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=min_pts).fit(data)
    distances, indices = nbrs.kneighbors(data)

    return distances

def min_max_normalize_df(df):
    return (df - df.min()) / (df.max() - df.min())

def min_max_normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def reverse_min_max_normalize_array(normalized_arr, arr):
    return (normalized_arr * (np.max(arr) - np.min(arr))) + np.min(arr)
