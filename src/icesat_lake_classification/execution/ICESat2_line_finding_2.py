import os

import numpy as np
import pandas as pd
from scipy.stats import mode

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth

if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    base_dir = 'F:/onderzoeken/thesis_msc/'
    write_full_df = True
    write_only_class_df = True
    plot_lakes = True

    classification_dir = 'F:/onderzoeken/thesis_msc/Exploration/data/cluster'
    exploration_data_dir = 'F:/onderzoeken/thesis_msc/Exploration/data/'
    classification_df_fn_list = pth.get_files_from_folder(classification_dir, '*1222*gt*l*_class.csv')

    ### Parameters
    # Step 2
    window_surface_line1 = 50

    buffer_step2_line1 = 0.75

    # Step 3
    window_bottom_line = 50
    buffer_bottom_line = - 0.4

    # step lake classification
    window_lake_class = 10

    # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4
    for fn in classification_df_fn_list:

        utl.log("Loading classification track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('loading data'):
            classification_df = pd.read_csv(fn, usecols=['lon', 'lat', 'height', 'distance','clusters']) #, encoding='latin-1')
            surface_df = classification_df.loc[classification_df['clusters'] == 2]

        utl.log("extracting surface for: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('surface line creation'):
            # make rolling average of surface and calculate difference with normal data

            surface_df.sort_values('distance', inplace=True)
            surface_line = pd.DataFrame(utl.rollBy(surface_df['height'], surface_df['distance'], window_surface_line1, np.nanmedian))
            result_index = [utl.find_nearest_sorted(surface_df['distance'].values, value) for value in surface_line.index]

            surface_line['idx'] = surface_df['distance'].iloc[result_index].index
            surface_line = surface_line.reset_index().set_index('idx')
            surface_line.rename(columns={'index': 'distance', 0: 'height'}, inplace=True)
            surface_line['distance'] = surface_line['distance'] + (window_surface_line1/2)

            surface_df_window = utl.interpolate_df_to_new_index(surface_df,
                                                        surface_line.loc[:, ['distance', 'height']].copy(), 'distance')

            df_interp = utl.interpolate_df_to_new_index(classification_df,
                                                        surface_df_window.loc[:, ['distance', 'height']].copy(),
                                                        'distance')
            del(surface_df, surface_df_window, surface_line)


        ### step 3 - Extracting the bottom
        utl.log("Extracting bottom for: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('bottom line creation'):

            df_interp['diff'] = classification_df['height'] - df_interp['height']
            bottom_df = classification_df.loc[(classification_df['clusters'] == 3) & (df_interp['diff'] < buffer_bottom_line)]
            classification_df['clusters'].iloc[np.where((classification_df['clusters'] == 3) & (df_interp['diff'] < buffer_bottom_line))] = 5

            bottom_df.sort_values('distance', inplace=True)
            bottom_line = pd.DataFrame(utl.rollBy(bottom_df['height'], bottom_df['distance'], window_bottom_line, np.nanmean))
            result_index = [utl.find_nearest_sorted(bottom_df['distance'].values, value - (window_bottom_line)) for value in bottom_line.index]

            bottom_line['idx'] = bottom_df['distance'].iloc[result_index].index
            bottom_line = bottom_line.reset_index().set_index('idx')
            bottom_line.rename(columns={'index': 'distance', 0: 'height'}, inplace=True)
            bottom_line['distance'] = bottom_line['distance'] + (window_bottom_line / 2)

            bottom_df_window = utl.interpolate_df_to_new_index(bottom_df,
                                                        bottom_line.loc[:, ['distance', 'height']].copy(), 'distance')

            bottom_df_interp = utl.interpolate_df_to_new_index(classification_df,
                                                        bottom_df_window.loc[:, ['distance', 'height']].copy(), 'distance')

            del (bottom_df, bottom_df_window, bottom_line)


        utl.log('Sorting data and calculating slope and lake depth', log_level='INFO')
        with utl.codeTimer('making lake classification for bottom line'):
            # sort data on distance
            classification_df['bottom_distance'] = bottom_df_interp['distance'].copy()
            classification_df['surface_distance'] = df_interp['distance'].copy()
            classification_df['bottom_height'] = bottom_df_interp['height'].copy()
            classification_df['surface_height'] = df_interp['height'].copy()
            classification_df.sort_values(by=['distance'], inplace=True)
            del(df_interp, bottom_df_interp)

            classification_df['lake_diff'] = classification_df['surface_height'] - classification_df['bottom_height']

            classification_df['angle'] = np.rad2deg( np.arctan2(np.roll(classification_df['surface_height'], 1) - classification_df['surface_height'],
                                                                 np.roll(classification_df['surface_distance'], 1) - classification_df['surface_distance'])) #angle = np.rad2deg(np.arctan2(y[-1] - y[0], x[-1] - x[0]))
            classification_df['slope'] = np.abs((np.roll(classification_df['surface_height'], 1) - classification_df['surface_height']) / ( np.roll(classification_df['surface_distance'], 1) - classification_df['surface_distance'])) # abs(rise/run)

            classification_df['lake'] = np.where((classification_df['lake_diff'] > 1.5) & (np.abs(classification_df['slope']) < 0.0025), 1, 0)
            classification_df['lake'].iloc[np.where((classification_df['lake_diff'] > 1.5) & (np.abs(classification_df['slope']) >= 0.0025))] = 2
            classification_df['lake'].iloc[np.where((np.abs(classification_df['slope']) < 0.0025) & (classification_df['lake_diff'] < 1.5))] = 3

        utl.log('Creating bottom line lake classification - mode over window', log_level='INFO')
        with utl.codeTimer('roling window of lake line'):

            bottom_line_class = pd.DataFrame(utl.rollBy_mode(classification_df['lake'], classification_df['distance'], window_lake_class, mode))
            result_index = [utl.find_nearest_sorted(classification_df['distance'].values, value + (window_lake_class/2)) for value in bottom_line_class.index]

            bottom_line_class['idx'] = classification_df['distance'].iloc[result_index].index
            bottom_line_class = bottom_line_class.reset_index().set_index('idx')
            bottom_line_class.rename(columns={'index': 'distance', 0: 'lake'}, inplace=True)

            classification_df['lake_rolling'] = bottom_line_class['lake']
            classification_df['lake_rolling'] = classification_df['lake_rolling'].fillna(method='ffill')

            del(bottom_line_class)

        if write_full_df:
            utl.log("Saving classification result for track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
            classification_df.to_csv(os.path.join(exploration_data_dir, 'lake_class', (os.path.basename(fn)[:-4] + '.csv')))

        if write_only_class_df:
            utl.log("Saving classification result - ONLY CLASS - for track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
            out_df = classification_df[['lon', 'lat', 'clusters', 'lake_rolling']].iloc[::100].copy()
            out_df.to_csv(os.path.join(exploration_data_dir, 'lake_class', (os.path.basename(fn)[:-9] + '_only_classification.csv')))


