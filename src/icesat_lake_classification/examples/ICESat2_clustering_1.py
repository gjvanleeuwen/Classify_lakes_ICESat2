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
    figures_dir = os.path.join(base_dir, 'Exploration/figures')
    lake_dir = os.path.join(figures_dir, 'lake')

    classification_dir = 'F:/onderzoeken/thesis_msc/Exploration/data'
    classification_df_fn_list = pth.get_files_from_folder(classification_dir, '*1222*gt2l*')

    utl.log('processing these files:', log_level='INFO')
    for fn in classification_df_fn_list:
        utl.log(os.path.basename(fn), log_level='INFO')

    example_lakes = {'ID': [0, 1, 2, 3, 4, 5, 6],
                     'beam': ['gt2l', 'gt2l', 'gt2l', 'gt1r', 'gt1r', 'gt1l', 'gt3l'],
                     'lon': [(-49.05 ,-49.02), (-48.87, -48.84), (-48.38, -48.35), (-48.53, -48.50), (-48.45, -48.42), (-49.01, -48.98), (-48.79, -48.76)],
                     'lat': [(68.95, 69.04), (68.41, 68.50), (66.81, 66.90), (67.07, 67.17), (66.81, 66.90), (68.59, 68.68), (68.42, 68.51)]}

    from_csv = True
    complete_track = True

    ### Parameters
    ph_per_image = 50000

    # Step 1
    min_pts = 6
    eps_method = 'max'

    # Step 2
    min_pts_step2 = 3
    eps_method2 = 'max'
    strict_step2 = 6

    if complete_track:
        for fn in classification_df_fn_list:

            utl.log("Start classification for track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
            classification_df = pd.read_csv(fn)
            n_ph = len(classification_df)

            start_index_array = np.arange(0, n_ph, ph_per_image)
            end_index_array = np.append(np.arange(ph_per_image, n_ph, ph_per_image), n_ph - 1)
            classification_df['clusters'] = np.zeros(n_ph, dtype='int')

            for i, (ph_start, ph_end) in enumerate(zip(start_index_array, end_index_array)):

                index_slice = slice(ph_start,ph_end)
                utl.log("Classify slice {}/{}".format(int(ph_start/ph_per_image),len(start_index_array)), log_level='INFO')

                # create dataframe with just 2 variables
                data_df = classification_df[['distance', 'height']].iloc[index_slice]
                data_df['distance'] = data_df['distance'] - min(data_df['distance'])

                eps1_outpath = os.path.join(figures_dir,os.path.basename(fn)[:-4],'histogram_EPS_ph_{}_method_{}_minpts_{}.png'.format(
                                                                 ph_start, eps_method, min_pts))
                eps1 = find_optimal_eps(data_df, min_pts=min_pts, method=eps_method,
                                        outpath=None, strict=1.5)

                clustering = DBSCAN(eps=eps1, min_samples=min_pts).fit(data_df)
                data_df['clusters'] = np.where(np.array(clustering.labels_) >= 0, 1, 0)  # signal = 1

                # outpath = os.path.join(figures_dir, os.path.basename(fn)[:-4],
                #                        'photon_classification_ph_{}_{}_{}.png'.format(ph_start,
                #                                                                          eps_method, min_pts))
                # plot_classified_photons(data_df, clusters, ph_start, eps1, outpath)

                ### step 2 --> eliminate surface
                data_df2 = data_df.loc[data_df['clusters'] > 0]
                noise_df = data_df.loc[data_df['clusters'] == 0]

                eps2_outpath = os.path.join(figures_dir, os.path.basename(fn)[:-4],'histogram_EPS_step2_ph_{}_method_{}_minpts_{}.png'.format(
                                                                 ph_start, eps_method2, min_pts_step2))
                eps2 = find_optimal_eps(data_df2, min_pts=min_pts_step2, method=eps_method2,
                                        outpath=None, strict=strict_step2)

                clustering = DBSCAN(eps=eps2, min_samples=min_pts_step2).fit(data_df2)
                data_df['clusters'][data_df['clusters'] > 0] = np.where(np.array(clustering.labels_) >= 0, 2, 3) # surface =2, bottom=3
                classification_df['clusters'].iloc[index_slice] = data_df['clusters']
                plt.close('all')

                # outpath2 = os.path.join(figures_dir, os.path.basename(fn)[:-4],
                #                        'photon_classification_ph_step2_{}_{}_{}.png'.format(ph_start,
                #                                                                          eps_method, min_pts))
                # plot_classified_photons(data_df, clusters, ph_start, eps1, outpath2)

            utl.log("Saving classification result for track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
            classification_df.to_csv(os.path.join(classification_dir, 'cluster', (os.path.basename(fn)[:-4] + '_class.csv')))







