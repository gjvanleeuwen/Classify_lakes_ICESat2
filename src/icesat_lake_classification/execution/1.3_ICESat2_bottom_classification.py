import os

import numpy as np
import pandas as pd
from scipy.stats import mode

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.raster_band import RasterBand
from icesat_lake_classification.ICESat2_data_management import add_surface_line, add_bottom_line, calc_ph_statistics, classify_lake

from osgeo import osr


# Define the Gaussian function
def box_and_ski (R, A_0, A_1, A_2):
    D = (A_0 / (R + A_1)) + A_2
    return D

if __name__ == "__main__":

    ## INFO ON WHAT TO PROCESS
    base_dir = 'F:/onderzoeken/thesis_msc/'
    s2_date = '20190617_L1C'
    file_mask = '*1222*gt2l*.csv'

    s2_band_list = ['NDWI_10m', 'B03', "B04", "B08", "B11", "B12"]
    s2_data_dir = "F:/onderzoeken/thesis_msc/data/Sentinel/{}".format(s2_date)
    overwrite_s2_train = False

    ### PARAMETERS
    refractive_index = 1.33
    NDWI_threshold = 0.21

    # SURFACE
    window_surface_line1 = 20  # meters
    window_surface_line1_sample = 5

    # BOTTOM
    window_bottom_line = 25  # meters
    window_bottom_line_sample = 5
    buffer_bottom_line = - 0.20  # meters

    # LAKE CLASS
    window_lake_class = 7.5  # meters
    lake_boundary = -1
    slope_boundary = 0.003

    ### PROCESSING
    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    figures_dir = os.path.join(base_dir, 'figures', s2_date)
    data_dir_in = os.path.join(base_dir, 'data')
    data_dir = os.path.join(base_dir, 'data', s2_date)

    cluster_fn_in_list = pth.get_files_from_folder(os.path.join(data_dir_in, 'cluster'), file_mask)

    depth_data_green_list, depth_data_red_list = [], []
    green_reflectance_list, red_reflectance_list = [], []
    # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, 5 = real bottom
    for fn in cluster_fn_in_list:

        utl.log("Loading classification track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('loading data'):
            classification_df = pd.read_csv(fn, usecols=['lon', 'lat', 'height', 'distance','clusters', 'dem']) #, encoding='latin-1')


        utl.log("extracting surface for: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('surface line creation'):
            # make rolling average of surface and calculate difference with normal data

            classification_df = add_surface_line(classification_df, window_surface_line1, window_surface_line1_sample)


        ### step 3 - Extracting the bottom
        utl.log("Extracting bottom for: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('bottom line creation'):

            classification_df = add_bottom_line(classification_df, window_bottom_line, window_bottom_line_sample, buffer_bottom_line, refractive_index)


        utl.log("Calculating Statistics for {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('Statistics'):
            classification_df = calc_ph_statistics(classification_df, window_size=10000)


        utl.log('Sorting data and classifying lake', log_level='INFO')
        with utl.codeTimer('making lake classification for bottom line'):
            # sort data on distance
            classification_df.sort_values(by=['distance'], inplace=True)
            classification_df = classify_lake(classification_df, lake_boundary, slope_boundary)


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
            classification_df.drop(['lake', 'ph_depth', 'dem'], axis=1, inplace=True)


        utl.log("Extacting Sentinel Data for beam", log_level='INFO')
        with utl.codeTimer('s2 extraction'):
            if not pth.check_existence(os.path.join(data_dir, 'Training', pth.get_filname_without_extension(fn)+ '.h5'), overwrite=overwrite_s2_train):
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
                df.to_hdf(os.path.join(data_dir, 'Training', pth.get_filname_without_extension(fn)+ '.h5'), key='df', mode='w')


        with utl.codeTimer('Saving the Dataframes'):
            utl.log("Saving classification result for track/beam: {}".format(os.path.basename(fn)[:-4]),
                    log_level='INFO')
            classification_df.to_hdf(os.path.join(data_dir, 'classification', pth.get_filname_without_extension(fn) +'.h5' ), key='df', mode='w')








