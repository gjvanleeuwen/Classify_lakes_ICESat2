import os

import numpy as np

def calc_min_pts(SN1, SN2, m=1):
    denominator = 2*SN1 - SN2
    numerator = np.log(2*SN1/SN2)
    min_pts = denominator/numerator
    if min_pts < 3:
        min_pts = 3
    return min_pts

def calc_SN1(N1, h1, l, R_a):
    """
    N = Total photons both signal and noise
    h = Vertical range, difference between min and max height in the segment
    l = along track track range. difference between min and max along track distance
    R_a = Radius within to compute ex
    Returns
    -------

    """
    SN1 = (np.pi * (R_a**2) * N1) /(h1*l)
    return SN1

def calc_sn2(N2, h2, l, R_a):
    """
    N = Total photons both signal and noise
    h = Vertical range, difference between min and max height in the segment
    l = along track track range. difference beteen min and max along track distance
    R_a = Radius within to compute ex
    - 1.5 during day, 2.5 during night
    Returns
    -------

    """
    SN2 = (np.pi * (R_a**2) *N2)/(h2 *l)
    return SN2