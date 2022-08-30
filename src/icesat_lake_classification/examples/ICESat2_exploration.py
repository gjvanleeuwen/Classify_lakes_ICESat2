import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.ICESat2_data_management import load_ICESat2_ATL03_data, get_classification_data, get_geo_info_beam, get_cum_along_track_distance
from icesat_lake_classification.ICESat2_visualization import scatter_plot_map, plot_photon_height, plot_photon_height_single_beam, plot_lake




if __name__ == "__main__":
    utl.set_log_level(log_level='INFO')

    ### Tasks
    write_coordinates = False
    make_plots = False
    save_data = True

    base_dir = 'F:/onderzoeken/thesis_msc/'
    data_dir = 'F:/onderzoeken/thesis_msc/data/ICESat2'
    exploration_dir = 'F:/onderzoeken/thesis_msc/Exploration'
    classification_dir = 'F:/onderzoeken/thesis_msc/Exploration/data'

    track_fn_list = pth.get_files_from_folder(data_dir, '*ATL03*1222*.h5')
    utl.log(track_fn_list, log_level='INFO')

    example_lakes = {'ID': [0, 1, 2, 3, 4, 5, 6],
                     'beam': ['gt2l', 'gt2l', 'gt2l', 'gt1r', 'gt1r', 'gt1l', 'gt3l'],
                     'lon': [(-49.05,-49.02), (-48.87, -48.84), (-48.38, -48.35), (-48.53, -48.50), (-48.45, -48.42), (-49.01, -48.98), (-48.79, -48.76)],
                     'lat': [(68.95, 69.04), (68.41, 68.50), (66.81, 66.90), (67.07, 67.17), (66.81, 66.90), (68.59, 68.68), (68.42, 68.51)]}

    for fn in track_fn_list:
        utl.log('1. -- load ICESat ATL03 track {} - Create Classification dataset'.format(fn), log_level='INFO')
        file_info, track_data, track_attributes, beam_names = load_ICESat2_ATL03_data(fn)

        utl.log('2. -- retrieving along track distance for all photons in track {}'.format(fn), log_level='INFO')
        along_track_distance_dict = get_cum_along_track_distance(track_data)

        if save_data:
            utl.log('3. -- saving photon and geo data per beam for classification', log_level='INFO')
            classification_data = get_classification_data(track_data, along_track_distance_dict, file_info, outdir=classification_dir, overwrite=True) #classification_dir)

        # for gtx in sorted(beam_names):
        #
        #     n_seg, n_ph, coordinates = get_geo_info_beam(track_data[gtx])
        #
        #     if make_plots:
        #         if not os.path.isdir(os.path.join(base_dir, 'Exploration/figures/', os.path.basename(fn)[:-3], gtx)):
        #             os.mkdir(os.path.join(base_dir, 'Exploration/figures/', os.path.basename(fn)[:-3], gtx))
        #
        #         start_index_array = np.arange(0,n_ph,10000)
        #         end_index_array = np.append(np.arange(10000, n_ph, 10000),n_ph-1)
        #         for ph_start, ph_end in zip(start_index_array, end_index_array):
        #
        #             index_slice = slice(ph_start, ph_end)
        #
        #             utl.log('Create figure Track {} - Beam {} - Photons: {}'.format(file_info['rgt'], gtx, index_slice), log_level='INFO')
        #
        #             plt.ioff()
        #             plot_photon_height_single_beam(track_data, gtx, along_track_distance_dict, index=index_slice,
        #                                            coords=None, beams=None, outpath=os.path.join(base_dir, 'Exploration/figures/', os.path.basename(fn)[:-3], gtx, 'Photon_height_{}.png'.format(ph_start)))