import os
import math

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.classification_optimization import find_optimal_eps
from icesat_lake_classification.ICESat2_visualization import plot_classified_photons

if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    plt.ioff()
    pd.options.mode.chained_assignment = None  # default='warn'

    base_dir = 'F:/onderzoeken/thesis_msc/'

    classification_dir = 'F:/onderzoeken/thesis_msc/Exploration/data/cluster'
    classification_df_fn_list = pth.get_files_from_folder(classification_dir, '*1222*gt1l*_class.csv')

    ### Parameters
    # Step 2
    buffer_step2_line1 = 0.75
    window_surface_line1 = 250
    window_surface_line2 = 400
    min_periods_line2 = 50

    # Step 3
    window_bottom_line = 100
    min_periods_bottom_line = 15
    buffer_bottom_line = - 0.25

    # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4
    for fn in classification_df_fn_list:

        utl.log("Loading classification track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        classification_df = pd.read_csv(fn, usecols=['height', 'distance','clusters']) #, encoding='latin-1')
        surface_df = classification_df.loc[classification_df['clusters'] == 2]

        # make rolling average of surface and calculate difference with normal data
        surface_df_window = surface_df.rolling(window=window_surface_line1).median()
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
        df_interp = utl.interpolate_df_to_new_index(classification_df,
                                                    surface_df_window.loc[:, ['distance', 'height']].copy(),
                                                    'distance')
        # df_interp.rename(columns={'surface_data_buffer_height_window': "height"},inplace=True)
        df_interp['diff'] = classification_df['height'] - df_interp['height']
        bottom_df = classification_df.loc[(classification_df['clusters'] == 3) & (df_interp['diff'] < buffer_bottom_line)]

        bottom_df_window = bottom_df.rolling(window=window_bottom_line,min_periods=min_periods_bottom_line).mean()
        bottom_df_interp = utl.interpolate_df_to_new_index(classification_df,
                                                    bottom_df_window.loc[:, ['distance', 'height']].copy(),
                                                    'distance')

        # sort data on distance
        classification_df['bottom_distance'] = bottom_df_window['distance'].copy()
        classification_df['surface_distance'] = surface_df_window['distance'].copy()
        classification_df['bottom_height'] = bottom_df_window['height'].copy()
        classification_df['surface_height'] = surface_df_window['height'].copy()
        classification_df.sort_values(by=['distance'], inplace=True)



        classification_df['lake_diff'] = df_interp['height'] - bottom_df_interp['height']
        classification_df['angle'] = np.rad2deg( np.arctan2(np.roll(df_interp.height, 1) - df_interp.height,
                                                             np.roll(df_interp.distance, 1) - df_interp.distance)) #angle = np.rad2deg(np.arctan2(y[-1] - y[0], x[-1] - x[0]))
        classification_df['slope'] = (np.roll(df_interp.height, 1) - df_interp.height) / ( np.roll(df_interp.distance, 1) - df_interp.distance) # rise/run
        classification_df['lake'] = np.where((classification_df['lake_diff'] > 1.5) & (np.abs(classification_df['slope']) < 0.1), 1, 0)


        ### make some graphs
        plt.ioff()
        ph_per_image = 50000
        n_ph = len(classification_df)

        if not pth.check_existence(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10])):
            os.mkdir(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10]))

        start_index_array = np.arange(0, n_ph, ph_per_image)
        end_index_array = np.append(np.arange(ph_per_image, n_ph, ph_per_image), n_ph - 1)

        for i, (ph_start, ph_end) in enumerate(zip(start_index_array, end_index_array)):

            if i < 250:
                continue

            index_slice = slice(ph_start,ph_end)
            utl.log("Plotting slice {}/{}".format(int(ph_start/ph_per_image),len(start_index_array)), log_level='INFO')

            # create dataframe with just 2 variables
            data_df = classification_df[['distance', 'height', 'clusters', 'bottom_distance', 'bottom_height', 'surface_distance', 'surface_height', 'lake']].iloc[index_slice]
            # data_df['distance'] = data_df['distance'] - min(data_df['distance'])

            f1, ax1 = plt.subplots(figsize=(20, 20))
            ax1.scatter(data_df['distance'], data_df['height'], c=np.zeros(len(data_df['clusters'])), cmap='Set2', marker=',',
                        s=0.5)

            from matplotlib.collections import LineCollection
            c = ['red' if a==1 else 'green' for a in data_df['lake']]
            # lines = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(data_df.loc[:, 'bottom_distance'].iloc[:-1], data_df.loc[:, 'bottom_height'].iloc[:-1], data_df.loc[:, 'bottom_distance'].iloc[1:], data_df.loc[:, 'bottom_height'].iloc[1:])]
            points = np.array([bottom_df_interp.loc[:, 'distance'].iloc[index_slice], bottom_df_interp.loc[:, 'height'].iloc[index_slice]]).T.reshape(-1, 1, 2)
            lines = np.concatenate([points[:-1], points[1:]], axis=1)
            colored_lines = LineCollection(lines, colors=c, linewidths=(2,))
            ax1.add_collection(colored_lines)


            # ax1.plot(data_df.loc[:, 'bottom_distance'], data_df.loc[:, 'bottom_height'])
            ax1.plot(data_df.loc[:, 'surface_distance'], data_df.loc[:, 'surface_height'])

            ax1.set_title('classification gt1l' + "- for photons {}".format(ph_start))
            ax1.get_xaxis().set_tick_params(which='both', direction='in')
            ax1.get_yaxis().set_tick_params(which='both', direction='in')
            ax1.set_xlabel('distance')
            ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')

            outpath = os.path.join(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10],
                                   'FINAL_classification_MEAN_ph_{}.png'.format(ph_start)))
            plt.savefig(outpath)
            plt.close('all')
