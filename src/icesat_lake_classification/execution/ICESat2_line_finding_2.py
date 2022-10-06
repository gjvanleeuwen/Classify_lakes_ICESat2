import os

import numpy as np
import pandas as pd
from scipy.stats import mode

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.raster_band import RasterBand

from osgeo import osr


# Define the Gaussian function
def box_and_ski (R, A_0, A_1, A_2):
    D = (A_0 / (R + A_1)) + A_2
    return D

if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    base_dir = 'F:/onderzoeken/thesis_msc/'
    figures_dir = os.path.join(base_dir, 'figures')
    data_dir = os.path.join(base_dir, 'data')

    s2_band_list = ['NDWI_10m', 'B03_10', "B04_10"]
    s2_data_dir = "F:/onderzoeken/thesis_msc/data/Sentinel/20190620"

    write_full_df = True
    write_only_class_df = False

    plot = True
    process_lines = True

    extract_s2_data = True
    overwrite_s2_train = False

    cluster_fn_in_list = pth.get_files_from_folder(os.path.join(data_dir, 'cluster'), '*1222*gt*l*.csv')

    ### Parameters
    refractive_index = 1.33
    # Step 2
    window_surface_line1 = 20  # meters

    # Step 3
    window_bottom_line = 25  # meters
    buffer_bottom_line = - 0.20  # meters

    # step lake classification
    window_lake_class = 7.5  # meters
    lake_boundary = 1
    slope_boundary = 0.0025

    # plots
    plot_starter_index = 150
    ph_per_image = 50000
    NDWI_threshold = 0.21

    depth_data_green_list, depth_data_red_list = [], []
    green_reflectance_list, red_reflectance_list = [], []
    # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, 5 = real bottom
    for fn in cluster_fn_in_list:

        utl.log("Loading classification track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('loading data'):
            classification_df = pd.read_csv(fn, usecols=['lon', 'lat', 'height', 'distance','clusters', 'dem']) #, encoding='latin-1')

        if process_lines:

            utl.log("extracting surface for: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
            with utl.codeTimer('surface line creation'):
                # make rolling average of surface and calculate difference with normal data

                surface_df = classification_df.loc[classification_df['clusters'] == 2]
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

                # adjust height of bottom photons and change clusters further - calculate difference between surface and bottom photons
                df_interp['diff'] = classification_df['height'] - df_interp['height']
                bottom_index = np.where((classification_df['clusters'] == 3) & (df_interp['diff'] < buffer_bottom_line))
                #real height = surface - current_depth/1.33
                classification_df['height'].iloc[bottom_index] = df_interp['height'].iloc[bottom_index] -  \
                                                                 ((df_interp['height'].iloc[bottom_index] - classification_df['height'].iloc[bottom_index])/ refractive_index)
                classification_df['clusters'].iloc[np.where((classification_df['clusters'] == 3) & (df_interp['diff'] < buffer_bottom_line))] = 5

                # seperate bottom photons
                bottom_df = classification_df.iloc[bottom_index].copy()

                bottom_df.sort_values('distance', inplace=True)
                bottom_line = pd.DataFrame(utl.rollBy(bottom_df['height'], bottom_df['distance'], window_bottom_line, np.nanmedian))
                result_index = [utl.find_nearest_sorted(bottom_df['distance'].values, value - (window_bottom_line)) for value in bottom_line.index]

                bottom_line['idx'] = bottom_df['distance'].iloc[result_index].index
                bottom_line = bottom_line.reset_index().set_index('idx')
                bottom_line.rename(columns={'index': 'distance', 0: 'height'}, inplace=True)
                bottom_line['distance'] = bottom_line['distance'] + (window_bottom_line / 2)

                bottom_df_window = utl.interpolate_df_to_new_index(bottom_df,
                                                            bottom_line.loc[:, ['distance', 'height']].copy(), 'distance')

                bottom_df_interp = utl.interpolate_df_to_new_index(classification_df,
                                                            bottom_df_window.loc[:, ['distance', 'height']].copy(), 'distance')

                del (bottom_df, bottom_df_window, bottom_line, bottom_index)


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

                classification_df['lake'] = np.where((classification_df['lake_diff'] > lake_boundary) & (np.abs(classification_df['slope']) < slope_boundary), 1, 0)
                classification_df['lake'].iloc[np.where((classification_df['lake_diff'] > lake_boundary) & (np.abs(classification_df['slope']) >= slope_boundary))] = 2
                classification_df['lake'].iloc[np.where((np.abs(classification_df['slope']) < slope_boundary) & (classification_df['lake_diff'] < lake_boundary))] = 3

            utl.log('Creating bottom line lake classification - mode over window', log_level='INFO')
            with utl.codeTimer('roling window of lake line'):

                bottom_line_class = pd.DataFrame(utl.rollBy_mode(classification_df['lake'], classification_df['distance'], window_lake_class, mode, nodata=0))
                result_index = [utl.find_nearest_sorted(classification_df['distance'].values, value + (window_lake_class/2)) for value in bottom_line_class.index]

                bottom_line_class['idx'] = classification_df['distance'].iloc[result_index].index
                bottom_line_class = bottom_line_class.reset_index().set_index('idx')
                bottom_line_class.rename(columns={'index': 'distance', 0: 'lake'}, inplace=True)
                bottom_line_class = bottom_line_class[~bottom_line_class.index.duplicated(keep='first')]

                classification_df['lake_rolling'] = bottom_line_class['lake']
                classification_df['lake_rolling'] = classification_df['lake_rolling'].fillna(method='ffill')

                del(bottom_line_class, result_index)

        utl.log("Making beam statistics for classification", log_level='INFO')
        with utl.codeTimer('Statistics'):
            classification_df['SNR'], classification_df['SurfBottR'],classification_df['SurfNoiseR'], classification_df['range'], classification_df['slope_mean'] = 0, 0, 0, 0, 0
            classification_df['dem_diff'] = np.abs(classification_df['height'] - classification_df['surface_height'])

            # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, 5 is real bottom
            for i, (ph_start, ph_end) in enumerate(zip(np.arange(0, len(classification_df), 10000),
                                                       np.append(np.arange(10000, len(classification_df), 10000), len(classification_df) - 1))):
                index_slice = slice(ph_start, ph_end)

                n_noise_ph = classification_df['clusters'].iloc[index_slice].value_counts()[0] # 0 = noise
                n_signal_ph = len(classification_df) - n_noise_ph
                if 5 in classification_df['clusters'].iloc[index_slice].value_counts():
                    n_bottom_ph = classification_df['clusters'].iloc[index_slice].value_counts()[5]
                else:
                    n_bottom_ph = 1# bottom
                if 2 in classification_df['clusters'].iloc[index_slice].value_counts():
                    n_surface_ph = classification_df['clusters'].iloc[index_slice].value_counts()[2]
                else:
                    n_surface_ph = 1 # surface

                classification_df['SNR'].iloc[index_slice] = n_signal_ph/n_noise_ph # around 1600
                classification_df['SurfBottR'].iloc[index_slice] = n_surface_ph/n_bottom_ph # around 5...
                classification_df['SurfNoiseR'].iloc[index_slice] = n_surface_ph/n_noise_ph # around 5...
                classification_df['range'].iloc[index_slice] = np.abs(classification_df['height'].iloc[index_slice].max() - classification_df['height'].iloc[index_slice].min())
                classification_df['slope_mean'].iloc[index_slice] = classification_df['slope'].iloc[index_slice].mean()

                # utl.log("stats for Photons {} - SNR {}, SurfNoiseR {},  SurfBottR {} ".format(index_slice,n_signal_ph/n_noise_ph,
                #                                                                     n_surface_ph/n_noise_ph,  n_surface_ph/n_bottom_ph), log_level='INFO')

        utl.log("Extacting Sentinel Data for beam", log_level='INFO')
        with utl.codeTimer('s2 extraction'):
            if extract_s2_data:
                if not pth.check_existence(os.path.join(data_dir, 'Training', (os.path.basename(fn))), overwrite=overwrite_s2_train):
                    s2_dir_list = pth.get_files_from_folder(s2_data_dir, '*.SAFE')

                    utl.log('Loading Beam file: {}'.format(fn), log_level="INFO")
                    classification_smaller = classification_df[['lon', 'lat']].iloc[::100].copy()

                    for band in s2_band_list: classification_smaller[band] = np.nan

                    # loop through the various S2 scenes for this date
                    for i, subdir in enumerate(s2_dir_list):
                        s2_files = pth.get_sorted_s2_filelist(subdir, band_list=s2_band_list, recursive=True,
                                                              extension='*')
                        utl.log('Loading Sentinel image {}/{} -- name: {}'.format(i, len(s2_dir_list), subdir), log_level="INFO")
                        # loop through the different S2 Bands
                        for s2_fn, band in zip(s2_files, s2_band_list):

                            RB = RasterBand(s2_fn, check_file_existence=True)
                            srs = osr.SpatialReference()
                            srs.SetWellKnownGeogCS("WGS84")
                            proj = srs.ExportToWkt()
                            RB = RB.warp(projection=proj)
                            values, index = RB.get_values_at_coordinates(classification_smaller['lon'].values, classification_smaller['lat'].values)
                            if not RB.no_data_value:
                                index = index[0][np.where((values != 0) & (~np.isnan(values)))]
                                values = values[np.where((values != 0) & (~np.isnan(values)))]
                            else:
                                index = index[0][np.where((values != RB.no_data_value) & (~np.isnan(values)))]
                                values = values[np.where((values != RB.no_data_value) & (~np.isnan(values)))]

                            if len(values) > 0:
                                # utl.log('Icesat track overlays with Sentinel image {} - Adding data to dataframe'.format(band), log_level='INFO')
                                classification_smaller[band].iloc[index] = values.copy()
                                # print(min(values), max(values))
                            else:
                                utl.log('NO match found - Sentinel image {} - does not overlay ICESat Track'.format(band),log_level='INFO')
                                break

                    df = classification_df[['lon', 'lat']].copy()
                    for band in s2_band_list:
                        df[band] = classification_smaller[band].copy()
                        last_index = np.where(df[band] > 0)[0][-1]
                        df[band].fillna(method='ffill', inplace=True)
                        df[band].iloc[last_index:] = np.nan

                    utl.log("Saving s2_training data", log_level='INFO')
                    df.to_csv(os.path.join(data_dir, 'Training', (os.path.basename(fn))))


        # load sentinel 2 data
        s2_df = pd.read_csv(os.path.join(data_dir, 'Training', (os.path.basename(fn))))

        classification_df[['NDWI_10m', 'B03_10', 'B04_10']] = s2_df[['NDWI_10m', 'B03_10', 'B04_10']].copy()
        del s2_df
        classification_df['NDWI_class'] = 0  # nodata value
        classification_df['NDWI_class'] = np.where((classification_df['NDWI_10m'] > NDWI_threshold), 1,
                                                   2)  # 1 for lakes, 2 for no lake

        with utl.codeTimer('Saving the Dataframes'):
            if write_full_df:
                print(len(classification_df))
                utl.log("Saving classification result for track/beam: {}".format(os.path.basename(fn)[:-4]),
                        log_level='INFO')
                classification_df.to_csv(os.path.join(data_dir, 'classification', (os.path.basename(fn))))

            if write_only_class_df:
                utl.log("Saving classification result - ONLY CLASS - for track/beam: {}".format(
                    os.path.basename(fn)[:-4]), log_level='INFO')
                out_df = classification_df[['lon', 'lat', 'clusters', 'lake_rolling']].iloc[::100].copy()
                out_df.to_csv(os.path.join('F:/onderzoeken/thesis_msc/Exploration/classification',
                                           (pth.get_filname_without_extension(fn) + '_only_classification.csv')))

        #make the empirical relation
        utl.log('Retrieving data for the empirical relations', log_level='INFO')
        with utl.codeTimer('empirical'):
            # needs to have s2 values, classified as lake by ICESAT and by the NDWI
            empirical_index_green = np.where((classification_df['B04_10'] > 0) & (classification_df['NDWI_10m'] > 0.35)
                                             & (classification_df['lake_rolling'] != 0) & (classification_df['SurfNoiseR'] > 2)
                                             & (classification_df['dem_diff'] < 25) & (classification_df['range'] < 200) & (classification_df['slope_mean'] < 0.1))

            green_depth = (classification_df['surface_height'].iloc[empirical_index_green] - classification_df['bottom_height'].iloc[empirical_index_green]).values
            depth_index_green = np.where((~np.isnan(green_depth)) & (green_depth < 25) & (green_depth > 0))

            # needs to have s2 values, classified as lake by ICESAT and by the NDWI
            empirical_index_red = np.where((classification_df['B03_10'] > 0) & (classification_df['NDWI_10m'] > 0.35)
                                           & (classification_df['lake_rolling'] != 0) & (classification_df['SurfNoiseR'] > 2)
                                           & (classification_df['dem_diff'] < 25) & (classification_df['range'] < 200) & (classification_df['slope_mean'] < 0.1))
            red_depth = (classification_df['surface_height'].iloc[empirical_index_red] - classification_df['bottom_height'].iloc[empirical_index_red]).values
            depth_index_red = np.where((~np.isnan(red_depth)) & (red_depth < 25) & (red_depth > 0))

            depth_data_green_list += list(green_depth[depth_index_green])
            depth_data_red_list += list(red_depth[depth_index_red])

            green_reflectance_list += list(classification_df['B04_10'].iloc[empirical_index_green].values[depth_index_green])
            red_reflectance_list += list(classification_df['B03_10'].iloc[empirical_index_red].values[depth_index_red])

    utl.log('Saving data for the empirical relation', log_level='INFO')
    pd.DataFrame({'depth':depth_data_green_list, 'reflectance': green_reflectance_list}).to_csv(os.path.join(data_dir, 'empirical', pth.get_filname_without_extension(fn)[0:44] + '_green.csv'))
    pd.DataFrame({'depth': depth_data_red_list, 'reflectance': red_reflectance_list}).to_csv(os.path.join(data_dir, 'empirical', pth.get_filname_without_extension(fn)[0:44] + '_red.csv'))






