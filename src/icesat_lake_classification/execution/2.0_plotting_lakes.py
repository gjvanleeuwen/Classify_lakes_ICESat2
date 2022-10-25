import os
import math

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import icesat_lake_classification.validation as validation

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.ICESat2_visualization import get_confusion_matrix, get_confusion_matrix2


if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    plot_starter_index = 200
    ph_per_image = 25000
    empirical = True
    NDWI_threshold = 0.21

    s2_band_list = ['NDWI_10m', 'B03', "B04", "B08", "B11", "B12"]
    s2_date = '20190617_L1C'
    base_dir = 'F:/onderzoeken/thesis_msc/'
    figures_dir = os.path.join(base_dir, 'figures', s2_date)
    data_dir = os.path.join(base_dir, 'data', s2_date)

    if not pth.check_existence(os.path.join(figures_dir, 'final')):
        os.mkdir(os.path.join(figures_dir, 'final'))

    classification_df_fn_list = pth.get_files_from_folder(os.path.join(data_dir, 'classification'), '*1222*gt1l*.h5')

    if empirical:
        utl.log('Plotting the Empirical relations', log_level='INFO')
        empirical_df_red = pd.read_csv(pth.get_files_from_folder(os.path.join(data_dir, 'empirical'),'*1222*green.csv')[0])
        empirical_df_green = pd.read_csv(pth.get_files_from_folder(os.path.join(data_dir, 'empirical'),'*1222*red.csv')[0])
        parameters_green, parameters_red, parameters_green_physical, parameters_red_physical = validation.estimate_relations(empirical_df_green, empirical_df_red)

    for fn in classification_df_fn_list:

        if not pth.check_existence(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn))):
            os.mkdir(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn)))

        utl.log(fn, log_level='INFO')
        classification_df = pd.read_hdf(fn)
        n_ph = len(classification_df)
        s2_df = pd.read_hdf(os.path.join(data_dir, 'Training', pth.get_filname_without_extension(fn) + '.h5'))

        classification_df[s2_band_list] = s2_df[s2_band_list].copy()
        del s2_df

        classification_df['NDWI_class'] = 0  # nodata value
        classification_df['NDWI_class'] = np.where((classification_df['NDWI_10m'] > NDWI_threshold), 1,
                                                   2)  # 1 for lakes, 2 for no lake

        classification_df['valid_point_index'] = np.where(
            (classification_df['NDWI_10m'] > NDWI_threshold) & (classification_df['lake_rolling'] == 1) & (
                    classification_df['SurfBottR'] > 1) & (classification_df['SurfNoiseR'] > 2.5) & (
                    classification_df['SurfBottR'] < 10) & (classification_df['dem_diff'] < 400) & (
                    classification_df['range'] < 400) & (classification_df['slope_mean'] < 0.1), 1, 0)

        if empirical:
            utl.log('Calculating empircal/physical depth lines for the plot', log_level='INFO')

            classification_df = validation.calculate_depth_from_relations(classification_df,parameters_green, parameters_red,
                                                                          parameters_green_physical, parameters_red_physical)

        if not pth.check_existence(os.path.join(figures_dir, 'final')):
            os.mkdir(os.path.join(figures_dir, 'final'))

        if not pth.check_existence(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn))):
            os.mkdir(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn)))

        utl.log('making confusion_matrix', log_level='INFO')
        # make confusion matrix
        CM_tuple = get_confusion_matrix(classification_df)
        CM_tuple2 = get_confusion_matrix2(classification_df)

        utl.log('Making color map', log_level='INFO')
        # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, bottom sure = 5
        cluster_map = {1: 'darkgrey', 0: 'bisque', 2: 'cornflowerblue', 3: 'wheat', 4: 'red',
                       5: 'mediumaquamarine'}
        cluster_map = {1: 'darkgrey', 0: 'bisque', 2: 'darkgrey', 3: 'darkgrey', 4: 'darkgrey',
                       5: 'darkgrey'}
        classification_df['c_cluster'] = [cluster_map[x] if not math.isnan(x) else cluster_map[0] for x in
                                          classification_df['clusters']]

        # clusters - 1 Lake, 0 no lake, 2, deep enough but to steep, 3 not deep enough but flat
        result_map_bottom = {1: 'Indigo', 0: 'dimgrey', 2: 'darkviolet', 3: 'violet', np.nan: 'dimgrey'}
        classification_df['c_bottom'] = [result_map_bottom[x] if not math.isnan(x) else result_map_bottom[0] for
                                         x in classification_df['lake_rolling']]

        # clusters - 1 lake in NDWI, 0 is nodata, 2 no lake
        result_map_surface = {1: 'Indigo', 0: 'dimgrey', 2: 'lightgrey', np.nan: 'dimgrey'}
        result_map_surface = {1: 'dimgrey', 0: 'dimgrey', 2: 'lightgrey', np.nan: 'dimgrey'}
        classification_df['c_surface'] = [result_map_surface[x] if not math.isnan(x) else result_map_surface[0]
                                          for x in classification_df['NDWI_class']]

        n_ph = len(classification_df)
        start_index_array = np.arange(0, n_ph, ph_per_image)
        end_index_array = np.append(np.arange(ph_per_image, n_ph, ph_per_image), n_ph - 1)
        for i, (ph_start, ph_end) in enumerate(zip(start_index_array, end_index_array)):

            plt.ioff()
            utl.log("Iterating through track -  {}/{} - Photon count {}".format(int(ph_start / ph_per_image) + 1,
                                                                            len(start_index_array),
                                                                            ph_start), log_level='INFO')

            index_slice = slice(ph_start, ph_end)
            if (i > plot_starter_index) & ((1 in classification_df['NDWI_class'].iloc[index_slice].unique()) & (
                    1 in classification_df['lake_rolling'].iloc[index_slice].unique())):

                utl.log("Conditions apply - Plotting this graph- {}".format(
                    classification_df['NDWI_class'].iloc[index_slice].unique()), log_level='INFO')

                f1, ax1 = plt.subplots(figsize=(15, 12.5))
                ax1.scatter(classification_df['distance'].iloc[index_slice],
                            classification_df['height'].iloc[index_slice],
                            c=classification_df['c_cluster'].iloc[index_slice], marker=',', s=0.5)

                if empirical:
                    # plot empirical
                    ax1.plot(classification_df['surface_distance'].iloc[index_slice],
                             classification_df['red'].iloc[index_slice], 'mediumseagreen', label='Empirical - B03')
                    ax1.plot(classification_df['surface_distance'].iloc[index_slice],
                             classification_df['green'].iloc[index_slice], 'maroon' , label='Empirical - B04')

                    # plot physical
                    ax1.plot(classification_df['surface_distance'].iloc[index_slice],
                             classification_df['red_phys'].iloc[index_slice], 'tab:olive', label='Physical - B03')
                    ax1.plot(classification_df['surface_distance'].iloc[index_slice],
                             classification_df['green_phys'].iloc[index_slice], 'orangered' , label='Physical - B04')

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

                ax1.set_title('')
                ax1.get_xaxis().set_tick_params(which='both', direction='in')
                ax1.get_yaxis().set_tick_params(which='both', direction='in')
                ax1.set_xlabel('Along-track distance [m]')
                ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')
                ax1.legend(loc='upper right')

                outpath = os.path.join(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn),
                                                    'lake_classification_s2_lon_{}_lat_{}_ph_{}_distance_median.png'.format(
                                                        np.round(classification_df['lon'].iloc[index_slice].iloc[0], 2),
                                                        np.round(classification_df['lat'].iloc[index_slice].iloc[0], 2),
                                                        2, np.round(classification_df['distance'].iloc[
                                                                               index_slice].iloc[0], 2))))
                plt.savefig(outpath)
                plt.close('all')
