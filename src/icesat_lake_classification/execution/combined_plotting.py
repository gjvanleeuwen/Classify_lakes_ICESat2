import os
import math

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.ICESat2_visualization import get_confusion_matrix

def box_and_ski (R, A_0, A_1, A_2):
    D = (A_0 / (R + A_1)) + A_2
    return D

def physical_single_channel_red(R_w, A_d):
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
    z = (np.log(A_d - R_inf) - np.log(R_w - R_inf))/g
    return z

def physical_single_channel_green(R_w, A_d):
    R_inf = 0.04
    g = 0.1413
    z = (np.log(A_d - R_inf) - np.log(R_w - R_inf))/g
    return z



if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    plot_starter_index = 200
    ph_per_image = 25000

    base_dir = 'F:/onderzoeken/thesis_msc/'
    figures_dir = os.path.join(base_dir, 'figures')
    data_dir = os.path.join(base_dir, 'data')

    if not pth.check_existence(os.path.join(figures_dir, 'final')):
        os.mkdir(os.path.join(figures_dir, 'final'))

    classification_df_fn_list = pth.get_files_from_folder(os.path.join(data_dir, 'classification'), '*1222*gt3l*.csv')

    utl.log('Plotting the Empirical relations', log_level='INFO')
    empirical_df_green = pd.read_csv(pth.get_files_from_folder(os.path.join(data_dir, 'empirical'),'*1222*green.csv')[0])
    empirical_df_red =  pd.read_csv(pth.get_files_from_folder(os.path.join(data_dir, 'empirical'),'*1222*red.csv')[0])

    parameters_green, cov_green = curve_fit(box_and_ski, empirical_df_green['reflectance'],empirical_df_green['depth'],
                                                   p0=[1, 0, 1], maxfev=5000)
    parameters_red, cov_red = curve_fit(box_and_ski, empirical_df_red['reflectance'], empirical_df_red['depth'],
                                               p0=parameters_green, maxfev=50000)

    eps = 1 / 100
    parameters_green_physical, cov_green_physical = curve_fit(physical_single_channel_green, (empirical_df_green['reflectance']/10000).iloc[np.where(np.isfinite(physical_single_channel_green(empirical_df_green['reflectance']/10000, 0.55)))],
                                                              empirical_df_green['depth'].iloc[np.where(np.isfinite(physical_single_channel_green(empirical_df_green['reflectance']/10000, 0.55)))],
                                                          p0=0.55, bounds=[0.3, 0.55])
    parameters_red_physical, cov_red_physical = curve_fit(physical_single_channel_red, empirical_df_red['reflectance']/10000, empirical_df_red['depth'],
                                                          p0=0.2, bounds=[0.15, 0.35])

    from sklearn.metrics import r2_score
    # plot empirical relations to training data
    fig, ax = plt.subplots(1,1)
    ax.scatter(empirical_df_green['reflectance'], empirical_df_green['depth'], color='dimgrey', s=1, alpha=0.01, marker='+')
    ax.plot(np.arange(0, 6000), box_and_ski(np.arange(0, 6000), *parameters_green),
                'mediumseagreen', label='Empirical R^2 {}'.format(np.round(r2_score(empirical_df_green['depth'], box_and_ski(empirical_df_green['reflectance'], *parameters_green)), 2)))

    ax.plot(np.arange(0.5, 6000.5),
                physical_single_channel_green(np.arange(0, 6000)/10000, parameters_green_physical[0]),
                'tab:olive',  label='Physical R^2 {}'.format(np.round(r2_score(empirical_df_green['depth'].iloc[np.where(np.isfinite(physical_single_channel_green(empirical_df_green['reflectance']/10000, 0.55)))],
                                                                               physical_single_channel_green((empirical_df_green['reflectance']/10000).iloc[np.where(np.isfinite(physical_single_channel_green(empirical_df_green['reflectance']/10000, 0.55)))],
                                                                                                             *parameters_green_physical)), 2)))
    ax.set_xlabel('Green reflectance e^4')
    ax.set_ylabel('depth (m)')
    ax.set_xlim([0, 6000])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    # axs[0].set_title('Relations')
    plt.savefig('Empirical_fit_{}_green.png'.format('1222_20190617'))
    plt.close('all')

    fig, ax = plt.subplots(1)
    ax.scatter(empirical_df_red['reflectance'], empirical_df_red['depth'], color='tab:grey', s=1, alpha=0.01, marker='+')
    ax.plot(np.arange(1000, 8000),
                box_and_ski(np.arange(1000, 8000), *parameters_red),
                'maroon',  label='Empirical R^2 {}'.format(np.round(r2_score(empirical_df_red['depth'], box_and_ski(empirical_df_red['reflectance'], *parameters_red)), 2)))
    ax.plot(np.arange(1000.5, 8000.5),
                physical_single_channel_red(np.arange(1000.5, 8000.5)/10000, parameters_red_physical[0]),
                'orangered', label='Physical R^2 {}'.format(np.round(r2_score(empirical_df_red['depth'].iloc[np.where(np.isfinite(physical_single_channel_red(empirical_df_red['reflectance']/10000, 0.55)))],
                                                                               physical_single_channel_red((empirical_df_red['reflectance']/10000).iloc[np.where(np.isfinite(physical_single_channel_red(empirical_df_red['reflectance']/10000, 0.55)))],
                                                                                                             *parameters_red_physical)), 2)))
    ax.set_ylabel('depth (m)')
    ax.set_xlabel('red reflectance e^4)')
    ax.set_xlim([0, 6000])
    ax.set_ylim([0, 15])

    ax.legend(loc='upper right')
    ax.set_title('curve fit for Sentinel 2 red and green bands')
    plt.savefig('Empirical_fit_{}_red.png'.format('1222_20190617'))
    plt.close('all')


    for fn in classification_df_fn_list:

        if not pth.check_existence(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn))):
            os.mkdir(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn)))

        utl.log(fn, log_level='INFO')
        classification_df = pd.read_csv(fn)
        s2_df = pd.read_csv(os.path.join(data_dir, 'Training', (os.path.basename(fn))))

        ### make some graphs
        n_ph = len(classification_df)

        utl.log('Calculating empircal/physical depth lines for the plot', log_level='INFO')
        classification_df['green'] = classification_df['surface_height'].copy()
        classification_df['green'][(classification_df['B04_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)] = \
            classification_df['surface_height'][(classification_df['B04_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)].values - \
            box_and_ski(classification_df['B04_10'][(classification_df['B04_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)].values,
                        parameters_green[0], parameters_green[1], parameters_green[2])

        classification_df['red'] = classification_df['surface_height'].copy()
        classification_df['red'][(classification_df['B03_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)] = \
            classification_df['surface_height'][
                (classification_df['B03_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)].values - \
            box_and_ski(classification_df['B03_10'][(classification_df['B03_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)].values,
                parameters_red[0], parameters_red[1], parameters_red[2])

        #physical
        classification_df['green_phys'] = classification_df['surface_height'].copy()
        classification_df['green_phys'][(classification_df['B04_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)] = \
            classification_df['surface_height'][(classification_df['B04_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)].values - \
            physical_single_channel_green(classification_df['B04_10'][(classification_df['B04_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)].values,
                        *parameters_green_physical)

        classification_df['red_phys'] = classification_df['surface_height'].copy()
        classification_df['red_phys'][(classification_df['B03_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)] = \
            classification_df['surface_height'][
                (classification_df['B03_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)].values - \
            physical_single_channel_red(classification_df['B03_10'][(classification_df['B03_10'] > 0) & (classification_df['NDWI_10m'] > 0.21)].values,
                *parameters_red_physical)


        if not pth.check_existence(os.path.join(figures_dir, 'final')):
            os.mkdir(os.path.join(figures_dir, 'final'))

        if not pth.check_existence(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn))):
            os.mkdir(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn)))

        utl.log('making confusion_matrix', log_level='INFO')
        # make confusion matrix
        CM_tuple = get_confusion_matrix(classification_df)

        utl.log('Making color map', log_level='INFO')
        # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, bottom sure = 5
        cluster_map = {1: 'darkgrey', 0: 'bisque', 2: 'cornflowerblue', 3: 'wheat', 4: 'red',
                       5: 'mediumaquamarine'}
        classification_df['c_cluster'] = [cluster_map[x] if not math.isnan(x) else cluster_map[0] for x in
                                          classification_df['clusters']]

        # clusters - 1 Lake, 0 no lake, 2, deep enough but to steep, 3 not deep enough but flat
        result_map_bottom = {1: 'Indigo', 0: 'dimgrey', 2: 'darkviolet', 3: 'violet', np.nan: 'dimgrey'}
        classification_df['c_bottom'] = [result_map_bottom[x] if not math.isnan(x) else result_map_bottom[0] for
                                         x in classification_df['lake_rolling']]

        # clusters - 1 lake in NDWI, 0 is nodata, 2 no lake
        result_map_surface = {1: 'Indigo', 0: 'dimgrey', 2: 'lightgrey', np.nan: 'dimgrey'}
        classification_df['c_surface'] = [result_map_surface[x] if not math.isnan(x) else result_map_surface[0]
                                          for x in classification_df['NDWI_class']]

        n_ph = len(classification_df)
        start_index_array = np.arange(0, n_ph, ph_per_image)
        end_index_array = np.append(np.arange(ph_per_image, n_ph, ph_per_image), n_ph - 1)
        for i, (ph_start, ph_end) in enumerate(zip(start_index_array, end_index_array)):

            plt.ioff()
            utl.log(
                "Iterating through track -  {}/{} - Photon count {}".format(int(ph_start / ph_per_image) + 1,
                                                                            len(start_index_array),
                                                                            ph_start), log_level='INFO')

            index_slice = slice(ph_start, ph_end)
            if (i > plot_starter_index) & ((1 in classification_df['NDWI_class'].iloc[index_slice].unique()) & (
                    1 in classification_df['lake_rolling'].iloc[index_slice].unique())):

                utl.log("Conditions apply - Plotting this graph- {}".format(
                    classification_df['NDWI_class'].iloc[index_slice].unique()), log_level='INFO')

                f1, ax1 = plt.subplots(figsize=(20, 20))
                ax1.scatter(classification_df['distance'].iloc[index_slice],
                            classification_df['height'].iloc[index_slice],
                            c=classification_df['c_cluster'].iloc[index_slice], marker=',', s=0.5)

                # plot empirical
                ax1.plot(classification_df['surface_distance'].iloc[index_slice],
                         classification_df['red'].iloc[index_slice], 'maroon')
                ax1.plot(classification_df['surface_distance'].iloc[index_slice],
                         classification_df['green'].iloc[index_slice], 'mediumseagreen')

                # plot physical
                ax1.plot(classification_df['surface_distance'].iloc[index_slice],
                         classification_df['red_phys'].iloc[index_slice], 'orangered')
                ax1.plot(classification_df['surface_distance'].iloc[index_slice],
                         classification_df['green_phys'].iloc[index_slice], 'tab:olive')

                # plot bottom_line
                points = np.array([classification_df['bottom_distance'].iloc[index_slice],
                                   classification_df['bottom_height'].iloc[index_slice]]).T.reshape(-1, 1, 2)
                lines = np.concatenate([points[:-1], points[1:]], axis=1)
                colored_lines = LineCollection(lines, colors=classification_df['c_bottom'].iloc[index_slice],
                                               linewidths=(2,))
                ax1.add_collection(colored_lines)

                # plot surface line
                points_surface = np.array([classification_df['surface_distance'].iloc[index_slice],
                                           classification_df['surface_height'].iloc[index_slice]]).T.reshape(-1, 1, 2)
                lines_surface = np.concatenate([points_surface[:-1], points_surface[1:]], axis=1)
                colored_lines_surface = LineCollection(lines_surface,
                                                       colors=classification_df['c_surface'].iloc[index_slice],
                                                       linewidths=(2,))
                ax1.add_collection(colored_lines_surface)

                # ax1.plot(classification_df['distance'].iloc[index_slice],
                #          classification_df['dem'].iloc[index_slice], 'r--')

                ax1.set_title('classification gt1l' + "- for photons {}".format(ph_start))
                ax1.get_xaxis().set_tick_params(which='both', direction='in')
                ax1.get_yaxis().set_tick_params(which='both', direction='in')
                ax1.set_xlabel('distance')
                ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')

                outpath = os.path.join(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn),
                                                    'lake_classification_s2_lon_{}_lat_{}_ph_{}_distance_median_bottom.png'.format(
                                                        np.round(classification_df['lon'].iloc[index_slice].iloc[0], 2),
                                                        np.round(classification_df['lat'].iloc[index_slice].iloc[0], 2),
                                                        ph_start, np.round(classification_df['distance'].iloc[
                                                                               index_slice].iloc[0], 2))))
                plt.savefig(outpath)
                plt.close('all')

            else:
                continue