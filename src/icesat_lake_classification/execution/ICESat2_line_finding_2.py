import os
import math

import numpy as np
import pandas as pd
from scipy.stats import mode
from numpy.lib.stride_tricks import sliding_window_view as window

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth

if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    plt.ioff()
    pd.options.mode.chained_assignment = None  # default='warn'

    base_dir = 'F:/onderzoeken/thesis_msc/'
    write_df = False
    plot_lakes = True

    classification_dir = 'F:/onderzoeken/thesis_msc/Exploration/data/cluster'
    exploration_data_dir = 'F:/onderzoeken/thesis_msc/Exploration/data/'
    classification_df_fn_list = pth.get_files_from_folder(classification_dir, '*1222*gt*l*_class.csv')

    ### Parameters
    # Step 2
    window_surface_line1 = 100
    min_periods_line1 = 100

    buffer_step2_line1 = 0.75
    window_surface_line2 = 400
    min_periods_line2 = 50

    # Step 3
    window_bottom_line = 50
    min_periods_bottom_line = 15
    buffer_bottom_line = - 0.4

    # step lake classification
    window_lake_class = 100

    # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4
    for fn in classification_df_fn_list:

        utl.log("Loading classification track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('loading data'):
            classification_df = pd.read_csv(fn, usecols=['lon', 'lat', 'height', 'distance','clusters']) #, encoding='latin-1')
            surface_df = classification_df.loc[classification_df['clusters'] == 2]

        utl.log("extracting surface for: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('surface line creation'):
            # make rolling average of surface and calculate difference with normal data
            surface_df_window = surface_df.rolling(window=window_surface_line1,  min_periods=min_periods_line1).median()
            surface_df_window = surface_df_window.rolling(window_surface_line1, min_periods_line1).mean()
            surface_df_window['diff'] = np.abs(surface_df_window['height'] - surface_df['height'])

            # Create a 2nd line even more smooth
            surface_df_window['surface_data_buffer_height'] = surface_df_window['height'].loc[
                surface_df_window['diff'] < (buffer_step2_line1)]
            surface_df_window['surface_data_buffer_height_window'] = surface_df_window[
                'surface_data_buffer_height'].copy().rolling(window=window_surface_line2,
                                                             min_periods=min_periods_line2).mean()
            surface_df_window['diff2'] = np.abs(
                surface_df_window['surface_data_buffer_height_window'] - surface_df['height'])

            # add detailed clusters to original dataframe for plot
            surface_df['clusters2'] = surface_df['clusters'].copy()
            surface_df['diff'] = surface_df_window['diff'].copy()
            surface_df['clusters2'].loc[surface_df['diff'] < (buffer_step2_line1)] = 4

        ### step 3 - Extracting the bottom
        utl.log("Extracting bottom for: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('bottom line creation'):
            df_interp = utl.interpolate_df_to_new_index(classification_df,
                                                        surface_df_window.loc[:, ['distance', 'height']].copy(),
                                                        'distance')
            # df_interp.rename(columns={'surface_data_buffer_height_window': "height"},inplace=True)
            df_interp['diff'] = classification_df['height'] - df_interp['height']
            bottom_df = classification_df.loc[(classification_df['clusters'] == 3) & (df_interp['diff'] < buffer_bottom_line)]
            classification_df['clusters'].iloc[np.where((classification_df['clusters'] == 3) & (df_interp['diff'] < buffer_bottom_line))] = 5

            bottom_df_window = bottom_df.rolling(window=window_bottom_line,min_periods=min_periods_bottom_line).mean()
            bottom_df_interp = utl.interpolate_df_to_new_index(classification_df,
                                                        bottom_df_window.loc[:, ['distance', 'height']].copy(),
                                                        'distance')

        utl.log('Sorting data and calculating slope and lake depth', log_level='INFO')
        with utl.codeTimer('mkaing lake classification for bottom line'):
            # sort data on distance
            classification_df['bottom_distance'] = bottom_df_window['distance'].copy()
            classification_df['surface_distance'] = surface_df_window['distance'].copy()
            classification_df['bottom_height'] = bottom_df_window['height'].copy()
            classification_df['surface_height'] = surface_df_window['height'].copy()
            classification_df.sort_values(by=['distance'], inplace=True)

            del(bottom_df_window, surface_df_window, bottom_df, surface_df)

            classification_df['lake_diff'] = df_interp['height'] - bottom_df_interp['height']
            classification_df['angle'] = np.rad2deg( np.arctan2(np.roll(df_interp.height, 1) - df_interp.height,
                                                                 np.roll(df_interp.distance, 1) - df_interp.distance)) #angle = np.rad2deg(np.arctan2(y[-1] - y[0], x[-1] - x[0]))
            classification_df['slope'] = np.abs((np.roll(df_interp.height, 1) - df_interp.height) / ( np.roll(df_interp.distance, 1) - df_interp.distance)) # abs(rise/run)
            classification_df['lake'] = np.where((classification_df['lake_diff'] > 1.5) & (np.abs(classification_df['slope']) < 0.0025), 1, 0)
            classification_df['lake'].iloc[np.where((classification_df['lake_diff'] > 1.5) & (np.abs(classification_df['slope']) >= 0.0025))] = 2
            classification_df['lake'].iloc[np.where((np.abs(classification_df['slope']) < 0.0025) & (classification_df['lake_diff'] < 1.5))] = 3

        utl.log('Creating bottom line lake classification - mode over window', log_level='INFO')
        with utl.codeTimer('roling window of lake line'):

            x = window(classification_df['lake'].to_numpy(), window_shape=window_lake_class)[::100]
            end_index = len(classification_df) - (100 * len(x))
            classification_df['lake_rolling'] = np.concatenate((np.repeat(mode(x.T)[0],100), np.zeros(end_index))).astype('int')

        if write_df:
            utl.log("Saving classification result for track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
            out_df = classification_df[['lon', 'lat', 'clusters', 'lake_rolling']].iloc[::100].copy()
            out_df.to_csv(os.path.join(exploration_data_dir, 'lake_class', (os.path.basename(fn)[:-9] + '_only_classification.csv')))
            # classification_df.to_csv(os.path.join(exploration_data_dir, 'lake_class', (os.path.basename(fn)[:-4] + '.csv')))


        if plot_lakes:
            ### make some graphs
            utl.log('Make graphs of the calculated mode', log_level='INFO')
            plt.ioff()
            ph_per_image = 100000
            n_ph = len(classification_df)

            if not pth.check_existence(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10])):
                os.mkdir(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10]))

            start_index_array = np.arange(0, n_ph, ph_per_image)
            end_index_array = np.append(np.arange(ph_per_image, n_ph, ph_per_image), n_ph - 1)

            for i, (ph_start, ph_end) in enumerate(zip(start_index_array, end_index_array)):

                if i < 75:
                    continue

                index_slice = slice(ph_start,ph_end)
                utl.log("Plotting slice {}/{} - Photon count {}".format(int(ph_start/ph_per_image), len(start_index_array), ph_start), log_level='INFO')

                # create dataframe with just 2 variables
                data_df = classification_df[['lon', 'lat', 'distance', 'height', 'clusters', 'bottom_distance', 'bottom_height', 'surface_distance', 'surface_height', 'lake', 'lake_rolling']].iloc[index_slice]

                # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, bottom sure = 5
                cluster_map = {1: 'darkgrey', 0: 'bisque', 2: 'dodgerblue', 3: 'wheat', 4: 'red', 5: 'mediumseagreen'}
                c_cluster = [cluster_map[x] for x in data_df['clusters']]

                f1, ax1 = plt.subplots(figsize=(20, 20))
                ax1.scatter(data_df['distance'], data_df['height'], c=c_cluster, marker=',',
                            s=0.5)

                result_map = {1: 'red', 0: 'green', 2: 'orange', 3: 'yellow'}
                c_bottom = [result_map[x] for x in data_df['lake_rolling']]

                points = np.array([bottom_df_interp.loc[:, 'distance'].iloc[index_slice], bottom_df_interp.loc[:, 'height'].iloc[index_slice]]).T.reshape(-1, 1, 2)
                lines = np.concatenate([points[:-1], points[1:]], axis=1)
                colored_lines = LineCollection(lines, colors=c_bottom, linewidths=(2,))
                ax1.add_collection(colored_lines)

                ax1.plot(df_interp.loc[:, 'distance'].iloc[index_slice], df_interp.loc[:, 'height'].iloc[index_slice])

                ax1.set_title('classification gt1l' + "- for photons {}".format(ph_start))
                ax1.get_xaxis().set_tick_params(which='both', direction='in')
                ax1.get_yaxis().set_tick_params(which='both', direction='in')
                ax1.set_xlabel('distance')
                ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')

                outpath = os.path.join(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10],
                                       'FINAL_classification_lon_{}_lat_{}_ph_{}_distance_.png'.format(
                                           np.round(data_df['lon'].iloc[0],2), np.round(data_df['lat'].iloc[0],2),
                                           ph_start, np.round(data_df['distance'].iloc[0],2))))
                plt.savefig(outpath)
                plt.close('all')
