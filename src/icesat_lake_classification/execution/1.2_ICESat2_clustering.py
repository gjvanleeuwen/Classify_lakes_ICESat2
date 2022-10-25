import os

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.classification_optimization import find_optimal_eps
from icesat_lake_classification.ICESat2_visualization import plot_classified_photons


if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    plt.ioff()
    pd.options.mode.chained_assignment = None  # default='warn'

    base_dir = 'F:/onderzoeken/thesis_msc/'
    figures_dir = os.path.join(base_dir, 'figures')
    data_dir = os.path.join(base_dir, 'data')

    if not pth.check_existence(os.path.join(figures_dir, 'class')):
        os.mkdir(os.path.join(figures_dir, 'class'))

    in_fn_list = pth.get_files_from_folder(os.path.join(data_dir, 'ICESat2_csv'), '*1222*gt*')

    utl.log('processing these files:', log_level='INFO')
    for fn in in_fn_list:
        utl.log(os.path.basename(fn), log_level='INFO')

    plot = True
    save = False

    ### Parameters
    ph_per_image = 25000
    iteration_starter = 200

    # Step 1
    min_pts = 6
    eps_method = 'max'
    strict_step1 = 1

    # Step 2
    min_pts_step2 = 3
    eps_method2 = 'max'
    strict_step2 = 3


    for fn in in_fn_list:

        utl.log("Start classification for track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        classification_df = pd.read_csv(fn)
        n_ph = len(classification_df)

        start_index_array = np.arange(0, n_ph, ph_per_image)
        end_index_array = np.append(np.arange(ph_per_image, n_ph, ph_per_image), n_ph - 1)
        classification_df['clusters'] = np.zeros(n_ph, dtype='int')

        for i, (ph_start, ph_end) in enumerate(zip(start_index_array, end_index_array)):

            if i < iteration_starter:
                continue

            index_slice = slice(ph_start,ph_end)
            utl.log("Classify slice {}/{}".format(int(ph_start/ph_per_image),len(start_index_array)), log_level='INFO')

            # create dataframe with just 2 variables
            data_df = classification_df[['distance', 'height']].iloc[index_slice]
            data_df['distance'] = data_df['distance'] - min(data_df['distance'])

            if plot:
                if not pth.check_existence(os.path.join(figures_dir, 'class', os.path.basename(fn)[:-4])):
                    os.mkdir(os.path.join(figures_dir, 'class', os.path.basename(fn)[:-4]))

                eps1_outpath = os.path.join(figures_dir, 'class', os.path.basename(fn)[:-4],'{}_Histogram_step_1'.format(
                                                             ph_start))
            else:
                eps1_outpath = None
            eps1 = find_optimal_eps(data_df, min_pts=min_pts, method=eps_method,
                                    outpath=eps1_outpath, strict=strict_step1)

            clustering = DBSCAN(eps=eps1, min_samples=min_pts).fit(data_df)
            data_df['clusters'] = np.where(np.array(clustering.labels_) >= 0, 1, 0)  # signal = 1

            if plot:
                outpath = os.path.join(figures_dir, 'class', os.path.basename(fn)[:-4],
                                       '{}_classification_step_1.png'.format(ph_start))
                plot_classified_photons(data_df, data_df['clusters'], ph_start, eps1, outpath)
                plt.close('all')

            ### step 2 --> eliminate surface
            data_df2 = data_df.loc[data_df['clusters'] > 0]
            noise_df = data_df.loc[data_df['clusters'] == 0]

            if plot:
                eps2_outpath = os.path.join(figures_dir, 'class', os.path.basename(fn)[:-4],'{}_Histogram_step_2'.format(
                                                             ph_start))
            else:
                eps2_outpath = None
            eps2 = find_optimal_eps(data_df2, min_pts=min_pts_step2, method=eps_method2,
                                    outpath=eps2_outpath, strict=strict_step2)

            clustering = DBSCAN(eps=eps2, min_samples=min_pts_step2).fit(data_df2)
            data_df['clusters'][data_df['clusters'] > 0] = np.where(np.array(clustering.labels_) >= 0, 2, 3) # surface =2, bottom=3
            classification_df['clusters'].iloc[index_slice] = data_df['clusters']

            if plot:
                outpath2 = os.path.join(figures_dir, 'class', os.path.basename(fn)[:-4],
                                       '{}_classification_step_2.png'.format(ph_start))
                plot_classified_photons(data_df, data_df['clusters'], ph_start, eps2, outpath2)
                plt.close('all')

        if save:
            utl.log("Saving classification result for track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
            classification_df.to_csv(os.path.join(data_dir, 'cluster', (os.path.basename(fn))))







