import os

import numpy as np
import pandas as pd

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth


if __name__ == "__main__":

    s2_date = '20190617_L1C'
    s2_band_list = ['NDWI_10m', 'B03', "B04", "B08", "B11", "B12"]
    s2_data_dir = "F:/onderzoeken/thesis_msc/data/Sentinel/{}".format(s2_date)

    base_dir = 'F:/onderzoeken/thesis_msc/'
    overwrite_empirical = True

    # Parameters
    NDWI_threshold = 0.21

    figures_dir = os.path.join(base_dir, 'figures', s2_date)
    data_dir_in = os.path.join(base_dir, 'data')
    data_dir = os.path.join(base_dir, 'data', s2_date)

    # Process
    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    cluster_fn_in_list = pth.get_files_from_folder(os.path.join(data_dir_in, 'cluster'), '*1222*gt*l*.h5')
    print(cluster_fn_in_list)
    utl.log(cluster_fn_in_list, log_level='INFO')

    depth_data_list = []
    reflectance_list = [[] for band in s2_band_list]

    # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, 5 = real bottom
    for fn in cluster_fn_in_list:
        utl.log("Loading classification track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('loading data'):

            classification_df = pd.read_hdf(os.path.join(data_dir, 'classification', pth.get_filname_without_extension(fn) +'.h5' ))
            s2_df = pd.read_hdf(os.path.join(data_dir, 'Training', pth.get_filname_without_extension(fn) + '.h5'))

            classification_df[s2_band_list] = s2_df[s2_band_list].copy()
            del s2_df

            classification_df['NDWI_class'] = 0  # nodata value
            classification_df['NDWI_class'] = np.where((classification_df['NDWI_10m'] > NDWI_threshold), 1, 2)  # 1 for lakes, 2 for no lake

            classification_df['valid_point_index'] = np.where((classification_df['NDWI_10m'] > NDWI_threshold)& (classification_df['lake_rolling'] == 1) & (
                                                                          classification_df['SurfBottR'] > 1) & (classification_df['SurfNoiseR'] > 0.5) & (
                                                                          classification_df['SurfBottR'] < 10) & (classification_df['dem_diff'] < 400) & (
                                                                          classification_df['range'] < 400) & (classification_df['slope_mean'] < 0.1), 1, 0)
            # print(classification_df['valid_point_index'].value_counts())
            # classification_df['valid_point_index2'] = np.where((classification_df['NDWI_10m'] > NDWI_threshold) & (classification_df['lake_rolling'] == 1) & (
            #                                                               classification_df['SurfBottR'] > 1) & (classification_df['SurfNoiseR'] > 0.5) & (
            #                                                               classification_df['SurfBottR'] < 10), 1, 0)
            # print(classification_df['valid_point_index2'].value_counts())


        if not pth.check_existence(os.path.join(data_dir, 'empirical', pth.get_filname_without_extension(fn)[0:44] + '_green.csv'), overwrite_empirical):
            #make the empirical relation
            utl.log('Retrieving data for the empirical relations', log_level='INFO')
            with utl.codeTimer('empirical'):
                empirical_index= np.where((classification_df['B03'] > 0) & (classification_df['B04'] > 0) & (classification_df['B11'] > 0) &
                                          (classification_df['valid_point_index'] > 0))
                depth = -1 * classification_df['lake_diff'].iloc[empirical_index].values
                depth_index = np.where((~np.isnan(depth)) & (depth < 25) & (depth > 0))

                for i, band in enumerate(s2_band_list):
                    reflectance_list[i] += list(
                        classification_df[band].iloc[empirical_index].values[depth_index])
                depth_data_list += list(depth[depth_index])
        print(len(depth_data_list))


    utl.log('Saving data for the empirical relation', log_level='INFO')
    df_dict = {'depth': depth_data_list}
    for i, band in enumerate(s2_band_list):
        df_dict.update({band : reflectance_list[i]})

    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(data_dir, 'empirical', pth.get_filname_without_extension(fn)[0:44] + '.csv'))