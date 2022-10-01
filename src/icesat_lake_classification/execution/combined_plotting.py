import os
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth


if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    plot_starter_index = 150
    ph_per_image = 50000
    NDWI_threshold = 0.18
    B3_B4_threshold = 0.09

    base_dir = 'F:/onderzoeken/thesis_msc/'
    figures_dir = os.path.join(base_dir, 'figures')
    data_dir = os.path.join(base_dir, 'data')

    if not pth.check_existence(os.path.join(figures_dir, 'final')):
        os.mkdir(os.path.join(figures_dir, 'final'))

    classification_df_fn_list = pth.get_files_from_folder(os.path.join(data_dir, 'classification'), '*1222*gt*l*.csv')

    for fn in classification_df_fn_list:

        if not pth.check_existence(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn))):
            os.mkdir(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn)))

        utl.log(fn, log_level='INFO')
        classification_df = pd.read_csv(fn)

        last_index = np.where(classification_df['NDWI_10m'] > 0)[0][-1]
        classification_df['NDWI_10m'].fillna(method='ffill',inplace=True)
        classification_df['NDWI_10m'].iloc[last_index:] = np.nan

        last_index = np.where(classification_df['B03_10'] > 0)[0][-1]
        classification_df['B03_10'].fillna(method='ffill',inplace=True)
        classification_df['B03_10'].iloc[last_index:] = np.nan

        last_index = np.where(classification_df['B04_10'] > 0)[0][-1]
        classification_df['B04_10'].fillna(method='ffill',inplace=True)
        classification_df['B04_10'].iloc[last_index:] = np.nan

        classification_df['NDWI_class'] = 0 # nodata value
        classification_df['NDWI_class'] = np.where((classification_df['NDWI_10m'] > NDWI_threshold) &
                                                   ((classification_df['B03_10'] - classification_df['B04_10']) > B3_B4_threshold), 1, 2) # 1 for lakes, 2 for no lake

        # make confusion matrix
        classification_df['positives'] = np.where((classification_df['lake_rolling'] == 1) & (classification_df['NDWI_class'] == 1), 1,0)
        classification_df['positives'] = np.where((classification_df['lake_rolling'] == 1) & (classification_df['NDWI_class'] == 1), 1, 0)
        classification_df['positives'] = np.where( (classification_df['lake_rolling'] == 1) & (classification_df['NDWI_class'] == 1), 1, 0)
        classification_df['positives'] = np.where((classification_df['lake_rolling'] == 1) & (classification_df['NDWI_class'] == 1), 1, 0)

        ### make some graphs
        n_ph = len(classification_df)

        if not pth.check_existence(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10])):
            os.mkdir(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10]))

        start_index_array = np.arange(0, n_ph, ph_per_image)
        end_index_array = np.append(np.arange(ph_per_image, n_ph, ph_per_image), n_ph - 1)

        # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, bottom sure = 5
        cluster_map = {1: 'darkgrey', 0: 'bisque', 2: 'cornflowerblue', 3: 'wheat', 4: 'red', 5: 'mediumaquamarine'}
        classification_df['c_cluster'] = [cluster_map[x] if not math.isnan(x) else cluster_map[0] for x in classification_df['clusters']]

        # clusters - 1 Lake, 0 no lake, 2, deep enough but to steep, 3 not deep enough but flat
        result_map_bottom = {1: 'Indigo', 0: 'dimgrey', 2: 'darkviolet', 3: 'violet', np.nan: 'dimgrey'}
        classification_df['c_bottom'] = [result_map_bottom[x] if not math.isnan(x) else result_map_bottom[0] for x in classification_df['lake_rolling'] ]

        # clusters - 1 lake in NDWI, 0 is nodata, 2 no lake
        result_map_surface = {1: 'Indigo', 0: 'dimgrey', 2: 'lightgrey', np.nan: 'dimgrey'}
        classification_df['c_surface'] = [result_map_surface[x] if not math.isnan(x) else result_map_surface[0] for x in classification_df['NDWI_class']]

        for i, (ph_start, ph_end) in enumerate(zip(start_index_array, end_index_array)):

            plt.ioff()
            utl.log("Iterating through track -  {}/{} - Photon count {}".format(int(ph_start / ph_per_image)+1, len(start_index_array),
                                                                ph_start), log_level='INFO')

            index_slice = slice(ph_start, ph_end)
            if (i > plot_starter_index) & ((1 in classification_df['NDWI_class'].iloc[index_slice].unique()) | (1 in classification_df['lake_rolling'].iloc[index_slice].unique())):

                utl.log("Conditions apply - Plotting this graph- {}".format(classification_df['NDWI_class'].iloc[index_slice].unique()), log_level='INFO')

                f1, ax1 = plt.subplots(figsize=(20, 20))
                ax1.scatter(classification_df['distance'].iloc[index_slice], classification_df['height'].iloc[index_slice],
                            c=classification_df['c_cluster'].iloc[index_slice], marker=',', s=0.5)

                # plot bottom_line
                points = np.array([classification_df['bottom_distance'].iloc[index_slice], classification_df['bottom_height'].iloc[index_slice]]).T.reshape(-1, 1,2)
                lines = np.concatenate([points[:-1], points[1:]], axis=1)
                colored_lines = LineCollection(lines, colors=classification_df['c_bottom'].iloc[index_slice], linewidths=(2,))
                ax1.add_collection(colored_lines)

                # plot surface line
                points = np.array([classification_df['surface_distance'].iloc[index_slice], classification_df['surface_height'].iloc[index_slice]]).T.reshape(-1,1,2)
                lines = np.concatenate([points[:-1], points[1:]], axis=1)
                colored_lines = LineCollection(lines, colors=classification_df['c_surface'].iloc[index_slice], linewidths=(2,))
                ax1.add_collection(colored_lines)

                ax1.set_title('classification gt1l' + "- for photons {}".format(ph_start))
                ax1.get_xaxis().set_tick_params(which='both', direction='in')
                ax1.get_yaxis().set_tick_params(which='both', direction='in')
                ax1.set_xlabel('distance')
                ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')

                outpath = os.path.join(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn),
                                                    'lake_classification_s2_lon_{}_lat_{}_ph_{}_distance.png'.format(
                                                        np.round(classification_df['lon'].iloc[index_slice].iloc[0], 2),
                                                        np.round(classification_df['lat'].iloc[index_slice].iloc[0], 2),
                                                        ph_start, np.round(classification_df['distance'].iloc[index_slice].iloc[0], 2))))
                plt.savefig(outpath)
                plt.close('all')

            else:
                continue