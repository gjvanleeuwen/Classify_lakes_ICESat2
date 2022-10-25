import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import icesat_lake_classification.validation as validation
import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth


if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    NDWI_threshold = 0.21
    s2_date = '20190617_L1C'
    base_dir = 'F:/onderzoeken/thesis_msc/'
    s2_band_list = ['NDWI_10m', 'B03', "B04", "B08", "B11", "B12"]

    figures_dir = os.path.join(base_dir, 'figures', s2_date)
    data_dir = os.path.join(base_dir, 'data', s2_date)

    classification_df_fn_list = pth.get_files_from_folder(os.path.join(data_dir, 'classification'), '*1222*gt*l*.h5')

    ## Processing
    if not pth.check_existence(os.path.join(figures_dir, 'final')):
        os.mkdir(os.path.join(figures_dir, 'final'))

    utl.log('Plotting the Empirical relations', log_level='INFO')
    empirical_df_red = pd.read_csv(pth.get_files_from_folder(os.path.join(data_dir, 'empirical'),'*1222*green.csv')[0])
    empirical_df_green = pd.read_csv(pth.get_files_from_folder(os.path.join(data_dir, 'empirical'),'*1222*red.csv')[0])

    parameters_green, parameters_red, parameters_green_physical, parameters_red_physical = validation.estimate_relations(
        empirical_df_green, empirical_df_red)

    depth_data_green_list, depth_data_red_list = [], []
    depth_data_green_list_physical, depth_data_red_list_physical = [], []
    depth_data_ICESAT2_list = []

    for fn in classification_df_fn_list:
        utl.log("Loading classification track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('loading data'):

            classification_df = pd.read_hdf(os.path.join(data_dir, 'classification', pth.get_filname_without_extension(fn) +'.h5' ))
            s2_df = pd.read_hdf(os.path.join(data_dir, 'Training', pth.get_filname_without_extension(fn) + '.h5'))

            classification_df[s2_band_list] = s2_df[s2_band_list].copy()
            del s2_df

            classification_df['NDWI_class'] = 0  # nodata value
            classification_df['NDWI_class'] = np.where((classification_df['NDWI_10m'] > NDWI_threshold), 1, 2)  # 1 for lakes, 2 for no lake

            classification_df['valid_point_index'] = np.where((classification_df['NDWI_10m'] > NDWI_threshold)& (classification_df['lake_rolling'] == 1) & (
                                                                          classification_df['SurfBottR'] > 1) & (classification_df['SurfNoiseR'] > 2.5) & (
                                                                          classification_df['SurfBottR'] < 10) & (classification_df['dem_diff'] < 400) & (
                                                                          classification_df['range'] < 400) & (classification_df['slope_mean'] < 0.1), 1, 0)

        utl.log('Calculating empircal/physical depth lines for the plot', log_level='INFO')
        classification_df = validation.calculate_depth_from_relations (classification_df,
                                    parameters_green, parameters_red, parameters_green_physical,
                                    parameters_red_physical)

        # plot empirical relations to training data
        fig, ax = plt.subplots(1,1)
        ax.scatter(empirical_df_green['reflectance'], empirical_df_green['depth'], color='dimgrey', s=1, alpha=0.01, marker='+')
        ax.plot(np.arange(0, 6000), validation.box_and_ski(np.arange(0, 6000), *parameters_green),
                    'mediumseagreen', label='Empirical R^2 {}'.format(np.round(r2_score(empirical_df_green['depth'], validation.box_and_ski(empirical_df_green['reflectance'], *parameters_green)), 2)))

        ax.plot(np.arange(0.5, 6000.5),
                    validation.physical_single_channel_green(np.arange(0, 6000)/10000, parameters_green_physical[0]),
                    'tab:olive',  label='Physical R^2 {}'.format(np.round(r2_score(empirical_df_green['depth'].iloc[np.where(np.isfinite(validation.physical_single_channel_green(empirical_df_green['reflectance']/10000, 0.55)))],
                                                                                   validation.physical_single_channel_green((empirical_df_green['reflectance']/10000).iloc[np.where(np.isfinite(validation.physical_single_channel_green(empirical_df_green['reflectance']/10000, 0.55)))],
                                                                                                                 *parameters_green_physical)), 2)))
        ax.set_xlabel('Green reflectance e^4')
        ax.set_ylabel('depth (m)')
        ax.set_xlim([0, 6000])
        ax.set_ylim([0, 15])

        ax.legend(loc='upper right')
        # axs[0].set_title('Relations')
        plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_S2_{}_green.png'.format('1222_20190617', s2_date)))
        plt.close('all')

        fig, ax = plt.subplots(1)
        ax.scatter(empirical_df_red['reflectance'], empirical_df_red['depth'], color='tab:grey', s=1, alpha=0.01, marker='+')
        ax.plot(np.arange(0, 8000),
                    validation.box_and_ski(np.arange(0, 8000), *parameters_red),
                    'maroon',  label='Empirical R^2 {}'.format(np.round(r2_score(empirical_df_red['depth'], validation.box_and_ski(empirical_df_red['reflectance'], *parameters_red)), 2)))
        ax.plot(np.arange(0.5, 8000.5),
                    validation.physical_single_channel_red(np.arange(0.5, 8000.5)/10000, parameters_red_physical[0]),
                    'orangered', label='Physical R^2 {}'.format(np.round(r2_score(empirical_df_red['depth'].iloc[np.where(np.isfinite(validation.physical_single_channel_red(empirical_df_red['reflectance']/10000, 0.55)))],
                                                                                   validation.physical_single_channel_red((empirical_df_red['reflectance']/10000).iloc[np.where(np.isfinite(validation.physical_single_channel_red(empirical_df_red['reflectance']/10000, 0.55)))],
                                                                                                                 *parameters_red_physical)), 2)))
        ax.set_ylabel('depth (m)')
        ax.set_xlabel('red reflectance e^4)')
        ax.set_xlim([0, 6000])
        ax.set_ylim([0, 15])

        ax.legend(loc='upper right')
        ax.set_title('curve fit for Sentinel 2 red and green bands')
        plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_S2_{}_red.png'.format('1222_20190617', s2_date)))
        plt.close('all')

        depth_data_green_list += list(classification_df['surface_height'].iloc[classification_df['valid_point_index']] - classification_df['green'].iloc[classification_df['valid_point_index']])
        depth_data_red_list += list(classification_df['surface_height'].iloc[classification_df['valid_point_index']] - classification_df['red'].iloc[classification_df['valid_point_index']])
        depth_data_green_list_physical += list(classification_df['surface_height'].iloc[classification_df['valid_point_index']] - classification_df['green_phys'].iloc[classification_df['valid_point_index']])
        depth_data_red_list_physical += list(classification_df['surface_height'].iloc[classification_df['valid_point_index']] - classification_df['red_phys'].iloc[classification_df['valid_point_index']])
        depth_data_ICESAT2_list += list(classification_df['surface_height'].iloc[classification_df['valid_point_index']] - classification_df['bottom_height'].iloc[classification_df['valid_point_index']])

    depth_data_green_list_physical = np.array(depth_data_green_list_physical)
    depth_data_red_list_physical = np.array(depth_data_red_list_physical)
    depth_data_green_list = np.array(depth_data_green_list)
    depth_data_red_list = np.array(depth_data_red_list)
    depth_data_ICESAT2_list = np.array(depth_data_ICESAT2_list)


    # plot empirical relations to training data
    fig, ax = plt.subplots(1, 1)
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_red_list)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_red_list, color='mediumseagreen', s=1, alpha=1, marker='+',
               label='Empirical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_red_list[nan_index1]),2)))
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_red_list_physical)))
    ax.scatter(depth_data_ICESAT2_list+0.0001, depth_data_red_list_physical, color='tab:olive', s=1, alpha=1, marker='+',
               label='Physical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_red_list_physical[nan_index1]),2)))
    ax.set_xlabel('ICESat-2 depth (m)')
    ax.set_ylabel('Empirical/Physical S2 depth (m)')
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    # axs[0].set_title('Relations')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_red_ICESAT_S2_{}.png'.format('1222_20190617', s2_date)))
    plt.close('all')

    fig, ax = plt.subplots(1)
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_green_list, color='maroon', s=1, alpha=0.01, marker='+',
               label='Empirical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_green_list[nan_index1]),2)))
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list_physical)))
    ax.scatter(depth_data_ICESAT2_list, depth_data_green_list_physical, color='orangered', s=1, alpha=0.01, marker='+',
               label='Physical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], depth_data_green_list_physical[nan_index1]),2)))
    ax.set_xlabel('ICESat-2 depth (m)')
    ax.set_ylabel('Empirical/Physical S2 depth (m)')
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    # ax.set_title('curve fit for Sentinel 2 red and green bands')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_green_ICESAT_S2_{}.png'.format('1222_20190617', s2_date)))
    plt.close('all')


    #combined
    fig, ax = plt.subplots(1)
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list)))
    ax.scatter(depth_data_ICESAT2_list, (depth_data_green_list + depth_data_red_list)/2, color='maroon', s=1, alpha=0.01, marker='+',
               label='Empirical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], ((depth_data_green_list + depth_data_red_list)/2)[nan_index1]),2)))
    nan_index1 = np.where((~np.isnan(depth_data_ICESAT2_list)) & (~np.isnan(depth_data_green_list_physical)))
    ax.scatter(depth_data_ICESAT2_list, (depth_data_green_list_physical + depth_data_red_list_physical)/2, color='orangered', s=1, alpha=0.01, marker='+',
               label='Physical R^2 {}'.format(np.round(r2_score(depth_data_ICESAT2_list[nan_index1], ((depth_data_green_list_physical + depth_data_red_list_physical)/2)[nan_index1]),2)))
    ax.set_xlabel('ICESat-2 depth (m)')
    ax.set_ylabel('Empirical/Physical S2 depth (m)')
    ax.set_xlim([0, np.max(depth_data_ICESAT2_list)])
    ax.set_ylim([0, np.max((depth_data_green_list + depth_data_red_list)/2)])

    ax.legend(loc='upper right')
    # ax.set_title('curve fit for Sentinel 2 red and green bands')
    plt.savefig(os.path.join(figures_dir, 'Empirical_fit_{}_combined_ICESAT_S2_{}.png'.format('1222_20190617', s2_date)))
    plt.close('all')

