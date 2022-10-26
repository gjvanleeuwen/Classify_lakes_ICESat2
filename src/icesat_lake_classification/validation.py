import numpy as np
from scipy.optimize import curve_fit

import pandas
import requests, io
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import icesat_lake_classification.utils as utl


def box_and_ski (R, A_0, A_1, A_2):
    D = (A_0 / (R + A_1)) + A_2
    return D


def physical_single_channel_red (R_w, A_d):
    # g_red = 0.8304, green=0.1413
    """where Ad is the albedo of the lake bed,
    R∞ is the reflectance of optically deep water (>40 m),
    Rw is the observed water reflectance (TOA reflectance)
    and z is lake depth.
    The quantity g is a two-way attenuation coefficient that accounts for losses in both upward and downward directions including absorption and scattering
    """
    # A_d Green - 0.40, A_d red = 0.19
    R_inf = 0.04
    g = 0.8304
    z = (np.log(A_d - R_inf) - np.log(R_w - R_inf)) / g
    return z


def physical_single_channel_green (R_w, A_d):
    R_inf = 0.04
    g = 0.1413
    z = (np.log(A_d - R_inf) - np.log(R_w - R_inf)) / g
    return z


def calculate_depth_from_relations(classification_df,
                                    parameters_green, parameters_red, parameters_green_physical,
                                    parameters_red_physical, model, s2_band_list):
    # empirical
    classification_df['green'] = classification_df['surface_height'].copy()
    classification_df['green'][(classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)] = \
        classification_df['surface_height'][
            (classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
        box_and_ski(
            classification_df['B04'][(classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)].values,
            parameters_green[0], parameters_green[1], parameters_green[2])

    classification_df['red'] = classification_df['surface_height'].copy()
    classification_df['red'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)] = \
        classification_df['surface_height'][
            (classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
        box_and_ski(
            classification_df['B03'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].values,
            parameters_red[0], parameters_red[1], parameters_red[2])

    # physical
    classification_df['green_phys'] = classification_df['surface_height'].copy()
    classification_df['green_phys'][(classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)] = \
        classification_df['surface_height'][
            (classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
        physical_single_channel_green(classification_df['B04'][(classification_df['B04'] > 0) & (
                classification_df['NDWI_class'] == 1)].values / 10000,
                                                 *parameters_green_physical)

    classification_df['red_phys'] = classification_df['surface_height'].copy()
    classification_df['red_phys'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)] = \
        classification_df['surface_height'][
            (classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
        physical_single_channel_red(classification_df['B03'][(classification_df['B03'] > 0) & (
                classification_df['NDWI_class'] == 1)].values / 10000,
                                               *parameters_red_physical)

    # machine learning
    classification_df['ML'] = classification_df['surface_height'].copy()
    classification_df['ML'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)] = \
        model.predict(classification_df[s2_band_list][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].copy())


    return classification_df


def estimate_relations(empirical_df, s2_band_list):
    parameters_green, cov_green = curve_fit(box_and_ski, empirical_df["B03"], empirical_df['depth'],
                                            p0=[1, 0, 1], maxfev=20000, method='trf')
    parameters_red, cov_red = curve_fit(box_and_ski, empirical_df['B04'], empirical_df['depth'],
                                        p0=[1, 0, 1], maxfev=20000, method='trf')

    parameters_green_physical, cov_green_physical = curve_fit(physical_single_channel_green,
                                                              (empirical_df["B03"] / 10000).iloc[np.where(
                                                                  np.isfinite(physical_single_channel_green(
                                                                      empirical_df["B03"] / 10000, 0.55)))],
                                                              empirical_df['depth'].iloc[np.where(np.isfinite(
                                                                  physical_single_channel_green(
                                                                      empirical_df["B03"] / 10000, 0.55)))],
                                                              p0=0.55, bounds=[0.3, 0.55])
    parameters_red_physical, cov_red_physical = curve_fit(physical_single_channel_red,
                                                          (empirical_df['B04']/ 10000).iloc[np.where(
                                                              np.isfinite(physical_single_channel_red(
                                                                  empirical_df['B04']/ 10000, 0.55)))],
                                                          empirical_df['depth'].iloc[np.where(np.isfinite(
                                                              physical_single_channel_red(
                                                                  empirical_df['B04']/ 10000, 0.55)))],
                                                          p0=0.2, bounds=[0.15, 0.35])

    #machine learning
    # model = RandomForestRegressor(n_estimators=10, random_state=0)
    # # Fitting the Random Forest Regression model to the data
    # model.fit(empirical_df[s2_band_list], empirical_df['depth'])
    model = sm.OLS(empirical_df['depth'], empirical_df[s2_band_list]).fit()

    return parameters_green, parameters_red, parameters_green_physical, parameters_red_physical, model


def estimate_relations_CV(empirical_df_full, s2_band_list, folds=3):
    for i in range(folds):
        empirical_df, empirical_df_test = train_test_split(empirical_df_full, test_size=0.2, random_state=i)

        parameters_green, cov_green = curve_fit(box_and_ski, empirical_df["B03"], empirical_df['depth'],
                                                p0=[1, 0, 1], maxfev=20000, method='trf')
        parameters_red, cov_red = curve_fit(box_and_ski, empirical_df['B04'], empirical_df['depth'],
                                            p0=[1, 0, 1], maxfev=20000, method='trf')
        ## testing
        depth_test_green = box_and_ski(empirical_df_test['B03'], *parameters_green)
        rmse_green = float(format(np.sqrt(mean_squared_error(empirical_df_test['depth'], depth_test_green)), '.3f'))
        R2_green = float(format((r2_score(empirical_df_test['depth'], depth_test_green)), '.3f'))
        depth_test_red = box_and_ski(empirical_df_test['B04'], *parameters_red)
        rmse_red = float(format(np.sqrt(mean_squared_error(empirical_df_test['depth'], depth_test_red)), '.3f'))
        R2_red = float(format((r2_score(empirical_df_test['depth'], depth_test_red)), '.3f'))
        utl.log('Empirical -- Green: RMSE {} , r2 {} ||| Red: RMSE {}, r2 {}'.format(rmse_green, R2_green, rmse_red, R2_red), log_level='INFO')

        parameters_green_physical, cov_green_physical = curve_fit(physical_single_channel_green,
                                                                  (empirical_df["B03"] / 10000).iloc[np.where(
                                                                      np.isfinite(physical_single_channel_green(
                                                                          empirical_df["B03"] / 10000, 0.55)))],
                                                                  empirical_df['depth'].iloc[np.where(np.isfinite(
                                                                      physical_single_channel_green(
                                                                          empirical_df["B03"] / 10000, 0.55)))],
                                                                  p0=0.55, bounds=[0.3, 0.55])
        parameters_red_physical, cov_red_physical = curve_fit(physical_single_channel_red,
                                                              (empirical_df['B04']/ 10000).iloc[np.where(
                                                                  np.isfinite(physical_single_channel_red(
                                                                      empirical_df['B04']/ 10000, 0.55)))],
                                                              empirical_df['depth'].iloc[np.where(np.isfinite(
                                                                  physical_single_channel_red(
                                                                      empirical_df['B04']/ 10000, 0.55)))],
                                                              p0=0.2, bounds=[0.15, 0.35])

        ## testing
        depth_test_green_phys = physical_single_channel_green(empirical_df_test['B03'], *parameters_green_physical)
        rmse_green_phys = float(format(np.sqrt(mean_squared_error(empirical_df_test['depth'], depth_test_green_phys)), '.3f'))
        R2_green_phys = float(format((r2_score(empirical_df_test['depth'], depth_test_green_phys)), '.3f'))
        depth_test_red_phys = physical_single_channel_red(empirical_df_test['B04'], *parameters_red_physical)
        rmse_red_phys = float(format(np.sqrt(mean_squared_error(empirical_df_test['depth'], depth_test_red_phys)), '.3f'))
        R2_red_phys = float(format((r2_score(empirical_df_test['depth'], depth_test_red_phys)), '.3f'))
        utl.log('Physical -- Green: RMSE {} , r2 {} ||| Red: RMSE {}, r2 {}'.format(rmse_green_phys, R2_green_phys, rmse_red_phys,
                                                                                     R2_red_phys), log_level='INFO')

        #machine learning
        # model = RandomForestRegressor(n_estimators=10, random_state=0)
        # # Fitting the Random Forest Regression model to the data
        # model.fit(empirical_df[s2_band_list], empirical_df['depth'])

        model = sm.OLS(empirical_df['depth'], empirical_df[s2_band_list]).fit()

        ## testing
        depth_test = model.predict(empirical_df_test[s2_band_list])
        rmse = float(format(np.sqrt(mean_squared_error(empirical_df_test['depth'], depth_test)), '.3f'))
        R2 = float(format((r2_score(empirical_df_test['depth'], depth_test)), '.3f'))
        utl.log('Machine -- RMSE {} , r2 {}'.format(rmse,R2), log_level='INFO')


    return
