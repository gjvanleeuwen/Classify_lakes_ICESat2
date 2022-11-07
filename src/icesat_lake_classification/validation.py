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
    # classification_df['green'] = classification_df['surface_height'].copy() * np.nan
    # classification_df['green'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)] = \
    #     classification_df['surface_height'][
    #         (classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
    #     box_and_ski(
    #         classification_df['B03'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].values,
    #         parameters_green[0], parameters_green[1], parameters_green[2])
    #
    # classification_df['red'] = classification_df['surface_height'].copy() * np.nan
    # classification_df['red'][(classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)] = \
    #     classification_df['surface_height'][
    #         (classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
    #     box_and_ski(
    #         classification_df['B04'][(classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)].values,
    #         parameters_red[0], parameters_red[1], parameters_red[2])
    #
    # # physical
    # classification_df['green_phys'] = classification_df['surface_height'].copy() * np.nan
    # classification_df['green_phys'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)] = \
    #     classification_df['surface_height'][
    #         (classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
    #     physical_single_channel_green(classification_df['B03'][(classification_df['B03'] > 0) & (
    #             classification_df['NDWI_class'] == 1)].values / 10000,
    #                                              *parameters_green_physical)
    #
    # classification_df['red_phys'] = classification_df['surface_height'].copy() * np.nan
    # classification_df['red_phys'][(classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)] = \
    #     classification_df['surface_height'][
    #         (classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
    #     physical_single_channel_red(classification_df['B04'][(classification_df['B04'] > 0) & (
    #             classification_df['NDWI_class'] == 1)].values / 10000,
    #                                            *parameters_red_physical)

    # machine learning
    classification_df['ML'] = classification_df['surface_height'].copy() * np.nan
    classification_df['ML'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)] = \
        model.predict(classification_df[s2_band_list][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].copy())


    return classification_df

def calculate_depth_from_relations(classification_df,
                                    parameters_green, parameters_red, parameters_green_physical,
                                    parameters_red_physical, model, s2_band_list):
    # empirical
    # classification_df['green'] = classification_df['surface_height'].copy() * np.nan
    # classification_df['green'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)] = \
    #     classification_df['surface_height'][
    #         (classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
    #     box_and_ski(
    #         classification_df['B03'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].values,
    #         parameters_green[0], parameters_green[1], parameters_green[2])
    #
    # classification_df['red'] = classification_df['surface_height'].copy() * np.nan
    # classification_df['red'][(classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)] = \
    #     classification_df['surface_height'][
    #         (classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
    #     box_and_ski(
    #         classification_df['B04'][(classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)].values,
    #         parameters_red[0], parameters_red[1], parameters_red[2])
    #
    # # physical
    # classification_df['green_phys'] = classification_df['surface_height'].copy() * np.nan
    # classification_df['green_phys'][(classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)] = \
    #     classification_df['surface_height'][
    #         (classification_df['B03'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
    #     physical_single_channel_green(classification_df['B03'][(classification_df['B03'] > 0) & (
    #             classification_df['NDWI_class'] == 1)].values / 10000,
    #                                              *parameters_green_physical)
    #
    # classification_df['red_phys'] = classification_df['surface_height'].copy() * np.nan
    # classification_df['red_phys'][(classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)] = \
    #     classification_df['surface_height'][
    #         (classification_df['B04'] > 0) & (classification_df['NDWI_class'] == 1)].values - \
    #     physical_single_channel_red(classification_df['B04'][(classification_df['B04'] > 0) & (
    #             classification_df['NDWI_class'] == 1)].values / 10000,
    #                                            *parameters_red_physical)

    # machine learning
    classification_df['ML'] = classification_df['surface_height'].copy() * np.nan
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
                                                              p0=0.55, bounds=[0.3, 0.7])
    parameters_red_physical, cov_red_physical = curve_fit(physical_single_channel_red,
                                                          (empirical_df['B04']/ 10000).iloc[np.where(
                                                              np.isfinite(physical_single_channel_red(
                                                                  empirical_df['B04']/ 10000, 0.55)))],
                                                          empirical_df['depth'].iloc[np.where(np.isfinite(
                                                              physical_single_channel_red(
                                                                  empirical_df['B04']/ 10000, 0.55)))],
                                                          p0=0.2, bounds=[0.15, 0.6])

    #machine learning
    model = RandomForestRegressor(n_estimators=250, random_state=0)
    empirical_df = empirical_df.groupby('NDWI_10m', as_index=False).median()
    # Fitting the Random Forest Regression model to the data
    model.fit(empirical_df[s2_band_list], empirical_df['depth'])
    # model = sm.OLS(empirical_df['depth'], empirical_df[s2_band_list]).fit()

    return parameters_green, parameters_red, parameters_green_physical, parameters_red_physical, model


def estimate_relations_CV(empirical_df_full, s2_band_list, figures_dir,s2_date, folds=5, test_size=0.2, path_addition="test"):
    import os
    from sklearn.inspection import permutation_importance
    depth_data_green_list, depth_data_red_list = [], []
    depth_data_green_list_physical, depth_data_red_list_physical = [], []
    depth_data_ICESAT2_list, depth_data_ML_list = [], []
    print(len(empirical_df_full))

    for i in range(folds):
        empirical_df, empirical_df_test = train_test_split(empirical_df_full, test_size=test_size, random_state=i)

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
                                                                  p0=0.55, bounds=[0.3, 0.7])
        parameters_red_physical, cov_red_physical = curve_fit(physical_single_channel_red,
                                                              (empirical_df['B04']/ 10000).iloc[np.where(
                                                                  np.isfinite(physical_single_channel_red(
                                                                      empirical_df['B04']/ 10000, 0.55)))],
                                                              empirical_df['depth'].iloc[np.where(np.isfinite(
                                                                  physical_single_channel_red(
                                                                      empirical_df['B04']/ 10000, 0.55)))],
                                                              p0=0.2, bounds=[0.15, 0.6])

        ## testing
        depth_test_green_phys = physical_single_channel_green(empirical_df_test['B03']/10000, *parameters_green_physical)
        rmse_green_phys = float(format(np.sqrt(mean_squared_error(empirical_df_test['depth'], depth_test_green_phys)), '.3f'))
        R2_green_phys = float(format((r2_score(empirical_df_test['depth'], depth_test_green_phys)), '.3f'))
        depth_test_red_phys = physical_single_channel_red(empirical_df_test['B04']/10000, *parameters_red_physical)
        rmse_red_phys = float(format(np.sqrt(mean_squared_error(empirical_df_test['depth'], depth_test_red_phys)), '.3f'))
        R2_red_phys = float(format((r2_score(empirical_df_test['depth'], depth_test_red_phys)), '.3f'))
        utl.log('Physical -- Green: RMSE {} , r2 {} ||| Red: RMSE {}, r2 {}'.format(rmse_green_phys, R2_green_phys, rmse_red_phys,
                                                                                     R2_red_phys), log_level='INFO')

        #machine learning
        model = RandomForestRegressor(n_estimators=250, random_state=0)
        # Fitting the Random Forest Regression model to the data
        model.fit(empirical_df[s2_band_list], empirical_df['depth'])

        # model = sm.OLS(empirical_df['depth'], empirical_df[s2_band_list]).fit()

        ## testing
        depth_test = model.predict(empirical_df_test[s2_band_list])
        rmse = float(format(np.sqrt(mean_squared_error(empirical_df_test['depth'], depth_test)), '.3f'))
        R2 = float(format((r2_score(empirical_df_test['depth'], depth_test)), '.3f'))
        utl.log('Machine -- RMSE {} , r2 {}'.format(rmse,R2), log_level='INFO')

        depth_data_green_list += list(depth_test_green)
        depth_data_red_list += list(depth_test_red)
        depth_data_green_list_physical += list(depth_test_green_phys)
        depth_data_red_list_physical += list(depth_test_red_phys)
        depth_data_ICESAT2_list += list(empirical_df_test['depth'])
        depth_data_ML_list += list(depth_test)


    depth_data_green_list_physical = np.array(depth_data_green_list_physical)
    depth_data_red_list_physical = np.array(depth_data_red_list_physical)
    depth_data_green_list = np.array(depth_data_green_list)
    depth_data_red_list = np.array(depth_data_red_list)
    depth_data_ICESAT2_list = np.array(depth_data_ICESAT2_list)
    depth_data_ML_list = np.array(depth_data_ML_list)

    # sort = model.feature_importances_.argsort()
    plt.barh(s2_band_list, model.feature_importances_, color='royalblue')
    plt.xlabel("Feature Importance", fontsize=16)
    plt.savefig(os.path.join(figures_dir, 'feature_importance.png'))
    plt.close('all')

    # plot empirical relations to training data
    fig, ax = plt.subplots(1, 1)
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_red_list)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_red_list, color='maroon', s=1, alpha=1, marker='+',
               label='E red | R^2 {} - RMSE {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_red_list[nan_index1]),2),
                                                         float(format(np.sqrt(mean_squared_error(depth_data_ICESAT2_list[nan_index1],depth_data_red_list[nan_index1])), '.3f'))))
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_red_list_physical)))
    ax.scatter(depth_data_ICESAT2_list+0.0001, depth_data_red_list_physical, color='orangered', s=1, alpha=1, marker='+',
               label='E red | R^2 {} - RMSE {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_red_list_physical[nan_index1]),2),
                                                         float(format(np.sqrt(mean_squared_error(depth_data_ICESAT2_list[nan_index1],depth_data_red_list_physical[nan_index1])), '.3f'))))

    ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.set_xlabel('ICESat-2 depth (m)', fontsize=16)
    ax.set_ylabel('Empirical/Physical S2 depth (m)', fontsize=16)
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    # axs[0].set_title('Relations')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_red_ICESAT_S2_{}_{}.png'.format('1222_20190617', s2_date, path_addition)))
    plt.close('all')

    # green
    fig, ax = plt.subplots(1)
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_green_list, color='darkgreen', s=1, marker='+',
               label='E Green | R^2 {} - RMSE {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_green_list[nan_index1]),2),
                                                         float(format(np.sqrt(mean_squared_error(depth_data_ICESAT2_list[nan_index1],depth_data_green_list[nan_index1])), '.3f'))))
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list_physical)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_green_list_physical, color='mediumseagreen', s=1, marker='+',
               label='Ph Green | R^2 {} - RMSE {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_green_list_physical[nan_index1]),2),
                                                         float(format(np.sqrt(mean_squared_error(depth_data_ICESAT2_list[nan_index1],depth_data_green_list_physical[nan_index1])), '.3f'))))

    ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.set_xlabel('ICESat-2 depth (m)', fontsize=16)
    ax.set_ylabel('Empirical/Physical S2 depth (m)', fontsize=16)
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    # ax.set_title('curve fit for Sentinel 2 red and green bands')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_green_ICESAT_S2_{}_{}.png'.format('1222_20190617', s2_date, path_addition)))
    plt.close('all')


    ## Machine
    fig, ax = plt.subplots(1)
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_ML_list, color='royalblue', s=1, marker='+',
               label='Empirical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_ML_list[nan_index1]),2)))

    ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.set_xlabel('ICESat-2 depth (m)', fontsize=16)
    ax.set_ylabel('Empirical/Physical S2 depth (m)',fontsize=16)
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    # ax.set_title('curve fit for Sentinel 2 red and green bands')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_ML_ICESAT_S2_{}_{}.png'.format('1222_20190617', s2_date, path_addition)))
    plt.close('all')
    print(parameters_green_physical, parameters_red_physical)

    ## Machine + empirical
    fig, ax = plt.subplots(1)
    print(len(depth_data_ICESAT2_list))
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_ML_list, color='royalblue', s=1, marker='+',
               label='ML | R^2 {} - RMSE {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_ML_list[nan_index1]),2),
                                                    float(format(np.sqrt(mean_squared_error(depth_data_ICESAT2_list[nan_index1],depth_data_ML_list[nan_index1])), '.3f'))))
    ax.scatter(depth_data_ICESAT2_list, depth_data_red_list, color='maroon', s=1, alpha=1, marker='+',
               label='E red | R^2 {} - RMSE {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_red_list[nan_index1]),2),
                                                         float(format(np.sqrt(mean_squared_error(depth_data_ICESAT2_list[nan_index1],depth_data_red_list[nan_index1])), '.3f'))))
    ax.scatter(depth_data_ICESAT2_list, depth_data_green_list, color='darkgreen', s=1, marker='+',
               label='E green | R^2 {} - RMSE {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_green_list[nan_index1]),2),
                                                       float(format(np.sqrt(mean_squared_error(depth_data_ICESAT2_list[nan_index1],depth_data_green_list[nan_index1])), '.3f'))))
    # ax.scatter(depth_data_ICESAT2_list, (depth_data_green_list + depth_data_red_list)/2, color='dimgrey', s=1, marker='+',
    #            label='E Combined R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], ((depth_data_green_list + depth_data_red_list)/2)[nan_index1]),2)))

    ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.set_xlabel('ICESat-2 depth (m)', fontsize=16)
    ax.set_ylabel('Empirical/Physical S2 depth (m)', fontsize=16)
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    # ax.set_title('curve fit for Sentinel 2 red and green bands')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_ML_emp_ICESAT_S2_{}_{}.png'.format('1222_20190617', s2_date, path_addition)))
    plt.close('all')



    return



def estimate_relations2(empirical_df, figures_dir,s2_date, path_addition="test"):
    import os
    depth_data_green_list, depth_data_red_list = [], []
    depth_data_green_list_physical, depth_data_red_list_physical = [], []
    depth_data_ICESAT2_list = []

    parameters_green, cov_green = curve_fit(box_and_ski, empirical_df["B03"], empirical_df['depth'],
                                            p0=[1, 0, 1], maxfev=20000, method='trf')
    parameters_red, cov_red = curve_fit(box_and_ski, empirical_df['B04'], empirical_df['depth'],
                                        p0=[1, 0, 1], maxfev=20000, method='trf')
    ## testing
    depth_test_green = box_and_ski(empirical_df['B03'], *parameters_green)
    rmse_green = float(format(np.sqrt(mean_squared_error(empirical_df['depth'], depth_test_green)), '.3f'))
    R2_green = float(format((r2_score(empirical_df['depth'], depth_test_green)), '.3f'))
    depth_test_red = box_and_ski(empirical_df['B04'], *parameters_red)
    rmse_red = float(format(np.sqrt(mean_squared_error(empirical_df['depth'], depth_test_red)), '.3f'))
    R2_red = float(format((r2_score(empirical_df['depth'], depth_test_red)), '.3f'))
    utl.log('Empirical -- Green: RMSE {} , r2 {} ||| Red: RMSE {}, r2 {}'.format(rmse_green, R2_green, rmse_red, R2_red), log_level='INFO')

    parameters_green_physical, cov_green_physical = curve_fit(physical_single_channel_green,
                                                              (empirical_df["B03"] / 10000).iloc[np.where(
                                                                  np.isfinite(physical_single_channel_green(
                                                                      empirical_df["B03"] / 10000, 0.55)))],
                                                              empirical_df['depth'].iloc[np.where(np.isfinite(
                                                                  physical_single_channel_green(
                                                                      empirical_df["B03"] / 10000, 0.55)))],
                                                              p0=0.55, bounds=[0.3, 0.7])
    parameters_red_physical, cov_red_physical = curve_fit(physical_single_channel_red,
                                                          (empirical_df['B04']/ 10000).iloc[np.where(
                                                              np.isfinite(physical_single_channel_red(
                                                                  empirical_df['B04']/ 10000, 0.55)))],
                                                          empirical_df['depth'].iloc[np.where(np.isfinite(
                                                              physical_single_channel_red(
                                                                  empirical_df['B04']/ 10000, 0.55)))],
                                                          p0=0.2, bounds=[0.15, 0.6])

    ## testing
    depth_test_green_phys = physical_single_channel_green(empirical_df['B03']/10000, *parameters_green_physical)
    rmse_green_phys = float(format(np.sqrt(mean_squared_error(empirical_df['depth'], depth_test_green_phys)), '.3f'))
    R2_green_phys = float(format((r2_score(empirical_df['depth'], depth_test_green_phys)), '.3f'))
    depth_test_red_phys = physical_single_channel_red(empirical_df['B04']/10000, *parameters_red_physical)
    rmse_red_phys = float(format(np.sqrt(mean_squared_error(empirical_df['depth'], depth_test_red_phys)), '.3f'))
    R2_red_phys = float(format((r2_score(empirical_df['depth'], depth_test_red_phys)), '.3f'))
    utl.log('Physical -- Green: RMSE {} , r2 {} ||| Red: RMSE {}, r2 {}'.format(rmse_green_phys, R2_green_phys, rmse_red_phys,
                                                                                 R2_red_phys), log_level='INFO')

    depth_data_green_list += list(depth_test_green)
    depth_data_red_list += list(depth_test_red)
    depth_data_green_list_physical += list(depth_test_green_phys)
    depth_data_red_list_physical += list(depth_test_red_phys)
    depth_data_ICESAT2_list += list(empirical_df['depth'])


    depth_data_green_list_physical = np.array(depth_data_green_list_physical)
    depth_data_red_list_physical = np.array(depth_data_red_list_physical)
    depth_data_green_list = np.array(depth_data_green_list)
    depth_data_red_list = np.array(depth_data_red_list)
    depth_data_ICESAT2_list = np.array(depth_data_ICESAT2_list)



    # plot empirical relations to training data
    fig, ax = plt.subplots(1, 1)
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_green_list, color='darkgreen', s=1, alpha=0.05,
               label='Empirical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_green_list[nan_index1]), 2)))
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list_physical)))
    ax.scatter(depth_data_ICESAT2_list+0.0001, depth_data_green_list_physical, color='mediumseagreen', s=1, alpha=0.05,
               label='Physical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_green_list_physical[nan_index1]),2)))

    ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.set_xlabel('ICESat-2 depth (m)', fontsize=16)
    ax.set_ylabel('Empirical/Physical S2 depth (m)',fontsize=16)
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    # axs[0].set_title('Relations')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_green_ICESAT_S2_{}_{}.png'.format('1222_20190617', s2_date, path_addition)))
    plt.close('all')

    # red
    fig, ax = plt.subplots(1)
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_red_list)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_red_list, color='maroon', s=1, alpha=0.05,
               label='Empirical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_red_list[nan_index1]),2)))
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_red_list_physical)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_red_list_physical, color='orangered', s=1, alpha=0.05,
               label='Physical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_red_list_physical[nan_index1]),2)))

    ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.set_xlabel('ICESat-2 depth (m)', fontsize=16)
    ax.set_ylabel('Empirical/Physical S2 depth (m)', fontsize=16)
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 10])

    ax.legend(loc='upper right')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_red_ICESAT_S2_{}_{}.png'.format('1222_20190617', s2_date, path_addition)))
    plt.close('all')


    # #combined
    # fig, ax = plt.subplots(1)
    # nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list)))
    # ax.scatter(depth_data_ICESAT2_list, (depth_data_green_list + depth_data_red_list)/2, color='dimgrey', s=1, marker='+',
    #            label='Empirical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], ((depth_data_green_list + depth_data_red_list)/2)[nan_index1]),2)))
    # nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list_physical)))
    # ax.scatter(depth_data_ICESAT2_list, (depth_data_green_list_physical + depth_data_red_list_physical)/2, color='lightgrey', s=1, marker='+',
    #            label='Physical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], ((depth_data_green_list_physical + depth_data_red_list_physical)/2)[nan_index1]),2)))
    # ax.set_xlabel('ICESat-2 depth (m)')
    # ax.set_ylabel('Empirical/Physical S2 depth (m)')
    # ax.set_xlim([0, 15])
    # ax.set_ylim([0, 15])
    #
    # ax.legend(loc='upper right')
    # # ax.set_title('curve fit for Sentinel 2 red and green bands')
    # plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_combined_ICESAT_S2_{}.png'.format('1222_20190617', s2_date)))
    # plt.close('all')


    ## Machine + empirical
    fig, ax = plt.subplots(1)
    print(len(depth_data_ICESAT2_list))
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_red_list, color='darkgreen', s=1, alpha=1, marker='+',
               label='E Green R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_red_list[nan_index1]),2)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_green_list, color='maroon', s=1, marker='+',
               label='E Red R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_green_list[nan_index1]),2)))
    # ax.scatter(depth_data_ICESAT2_list, (depth_data_green_list + depth_data_red_list)/2, color='dimgrey', s=1, marker='+',
    #            label='E Combined R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], ((depth_data_green_list + depth_data_red_list)/2)[nan_index1]),2)))

    ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.set_xlabel('ICESat-2 depth (m)', fontsize=16)
    ax.set_ylabel('Empirical/Physical S2 depth (m)', fontsize=16)
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    # ax.set_title('curve fit for Sentinel 2 red and green bands')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_ML_emp_ICESAT_S2_{}_{}.png'.format('1222_20190617', s2_date, path_addition)))
    plt.close('all')




    #################           PLOTTING

    # plot empirical relations to training data - Green
    fig, ax = plt.subplots(1,1)
    ax.scatter(empirical_df['B03'], empirical_df['depth'], color='lightgrey', s=1, alpha=0.05)
    depth_calc = box_and_ski(np.arange(0, 6000), *parameters_green)
    ax.plot(np.arange(0, 6000), depth_calc, color='darkgreen',
            label='Empirical | R^2 {} - RMSE {}'.format(
                np.round(R2_green, 2), np.round(rmse_green, 2)))

    depth_calc = physical_single_channel_green(np.arange(0, 6000)/10000, *parameters_green_physical)
    ax.plot(np.arange(0, 6000), depth_calc, color='mediumseagreen',
            label='Ph a_d = {} | R^2 {} - RMSE {}'.format(np.round(parameters_green_physical[0],1),
                np.round(R2_green_phys, 2), np.round(rmse_green_phys, 2)))


    depth_calc = physical_single_channel_green(np.arange(0, 6000)/10000, *parameters_green_physical *1.33)
    depth_calc2 = physical_single_channel_green(empirical_df['B03']/10000, *parameters_green_physical *1.33)
    rmse_green_phys = float(format(np.sqrt(mean_squared_error(empirical_df['depth'], depth_calc2)), '.3f'))
    R2_green_phys = float(format((r2_score(empirical_df['depth'], depth_calc2)), '.3f'))

    ax.plot(np.arange(0, 6000), depth_calc, linestyle='--',  color='darkolivegreen',
            label='Ph a_d +33% | R^2 {} - RMSE {}'.format(
                np.round(R2_green_phys, 2), np.round(rmse_green_phys, 2)))

    depth_calc = physical_single_channel_green(np.arange(0, 6000)/10000, *(parameters_green_physical - parameters_green_physical*0.33))
    depth_calc2 = physical_single_channel_green(empirical_df['B03']/10000, *(parameters_green_physical - parameters_green_physical*0.33))
    rmse_green_phys = float(format(np.sqrt(mean_squared_error(empirical_df['depth'], depth_calc2)), '.3f'))
    R2_green_phys = float(format((r2_score(empirical_df['depth'], depth_calc2)), '.3f'))
    ax.plot(np.arange(0, 6000), depth_calc, linestyle='--',  color='yellowgreen',
            label='Ph a_d -33% | R^2 {} - RMSE {}'.format(
                np.round(R2_green_phys, 2), np.round(rmse_green_phys, 2)))

    ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.set_xlabel('Green reflectance e^4', fontsize=16)
    ax.set_ylabel('depth (m)', fontsize=16)
    ax.set_xlim([900, 6000])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_S2_{}_green_1.png'.format('1222_20190617', s2_date)))
    plt.close('all')

    #################           PLOTTING

    # plot empirical relations to training data - red
    fig, ax = plt.subplots(1,1)
    ax.scatter(empirical_df['B04'], empirical_df['depth'], color='lightgrey', s=1, alpha=0.05)
    depth_calc = box_and_ski(np.arange(0, 6000), *parameters_red)
    ax.plot(np.arange(0, 6000), depth_calc, color='maroon',
            label='Empirical | R^2 {} - RMSE {}'.format(
                np.round(R2_red, 2), np.round(rmse_red, 2)))

    depth_calc = physical_single_channel_red(np.arange(0, 6000)/10000, *parameters_red_physical)
    ax.plot(np.arange(0, 6000), depth_calc, color='orangered',
            label='Ph a_d= {} | R^2 {} - RMSE {}'.format(np.round(parameters_red_physical[0],1),
                np.round(R2_red_phys, 2), np.round(rmse_red_phys, 2)))


    depth_calc = physical_single_channel_red(np.arange(0, 6000)/10000, *parameters_red_physical *1.33)
    depth_calc2 = physical_single_channel_red(empirical_df['B04']/10000, *parameters_red_physical *1.33)
    rmse_red_phys = float(format(np.sqrt(mean_squared_error(empirical_df['depth'], depth_calc2)), '.3f'))
    R2_red_phys = float(format((r2_score(empirical_df['depth'], depth_calc2)), '.3f'))

    ax.plot(np.arange(0, 6000), depth_calc, linestyle='--', color='orange',
            label='Ph a_d +33% | R^2 {} - RMSE {}'.format(
                np.round(R2_red_phys, 2), np.round(rmse_red_phys, 2)))

    depth_calc = physical_single_channel_red(np.arange(0, 6000)/10000, *(parameters_red_physical - parameters_red_physical*0.33))
    depth_calc2 = physical_single_channel_red(empirical_df['B04']/10000, *(parameters_red_physical - parameters_red_physical*0.33))
    rmse_red_phys = float(format(np.sqrt(mean_squared_error(empirical_df['depth'], depth_calc2)), '.3f'))
    R2_red_phys = float(format((r2_score(empirical_df['depth'], depth_calc2)), '.3f'))
    ax.plot(np.arange(0, 6000), depth_calc, linestyle='--', color='gold',
            label='Ph a_d -33% | R^2 {} - RMSE {}'.format(
                np.round(R2_red_phys, 2), np.round(rmse_red_phys, 2)))

    ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=12)
    ax.set_xlabel('Red reflectance *10^4', fontsize=16)
    ax.set_ylabel('depth (m)', fontsize=16)
    ax.set_xlim([325, 5000])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_S2_{}_red_1.png'.format('1222_20190617', s2_date)))
    plt.close('all')

    return