import logging

import numpy as np
import pandas as pd

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


def interpolate_df_to_new_index(df_original, df_small, index_col):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=df_original.index)
    df_out.index.name = df_original.index.name

    for colname, col in df_small.iteritems():
        if colname == index_col:
            df_out[colname] = df_original[index_col]
        else:
            df_out[colname] = np.interp(df_original[index_col], df_small[index_col], col)

    return df_out


def closestDistanceBetweenLines (a0, a1, b0, b1, clampAll=False, clampA0=False, clampA1=False, clampB0=False,
                                 clampB1=False):
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)


            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom;
    t1 = detB / denom;

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, np.linalg.norm(pA - pB)

import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        log("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)), log_level='INFO')
        return value

    return wrapper

import timeit

class codeTimer:
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start)
        log('Code block' + self.name + ' took: ' + str(self.took) + ' seconds', log_level='INFO')