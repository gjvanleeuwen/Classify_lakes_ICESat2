import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering

import icesat_lake_classification.utils as utl
from icesat_lake_classification.ICESat2_data_management import load_ICESat2_ATL03_data, get_distance_photons, write_geolocation_to_txt, get_geoloaction_info_track, \
    get_DEM_from_photon_height, get_classification_data
from icesat_lake_classification.ICESat2_visualization import scatter_plot_map, plot_photon_height, plot_photon_height_single_beam, plot_lake


if __name__ == "__main__":
    __author__ = "gjvanleeuwen"
    __copyright__ = "gjvanleeuwen"
    __license__ = "MIT"

    utl.set_log_level(log_level='INFO')

    base_dir = 'F:/onderzoeken/thesis_msc/'
    data_fn = 'ATL03_20190617064249_12220303_005_01'
    datapath = 'F:/onderzoeken/thesis_msc/data/ICESat2/ATL03_20190617064249_12220303_005_01.h5'

    ###     data track info:
    product = 'ATL03'
    beam_list = ['gt1l'] #, 'gt2l']
    make_plots = False
    write_coordinates = False

    utl.log('gather photon heights in order', log_level='INFO')
    file_info, IS2_atl03_mds, IS2_atl03_attrs, IS2_atl03_beams = load_ICESat2_ATL03_data(datapath)

    if write_coordinates:
        for gtx in sorted(IS2_atl03_beams):
            write_geolocation_to_txt(IS2_atl03_mds[gtx], 'F:/onderzoeken/thesis_msc/Exploration/Coordinates/ATL03_20190617064249_12220303_005_01_beam_{}.csv'.format(gtx))

    # _, _ , coordinates_gt1l = get_geoloaction_info_track(IS2_atl03_mds['gt1l'])
    # scatter_plot_map(coordinates[:,0], coordinates[:,1], thin=10000)

    if make_plots:
        if not os.path.isdir(os.path.join(base_dir, 'Exploration/figures', data_fn)):
            os.mkdir(os.path.join(base_dir, 'Exploration/figures', data_fn))

    along_track_distance_PE = {}
    dem_PE = {}

    for gtx in sorted(IS2_atl03_beams):
        utl.log('Creating figure of photon heights for beam {}'.format(gtx), log_level='INFO')
        # -- data and attributes for beam gtx
        val = IS2_atl03_mds[gtx]

        along_track_distance_PE.update({gtx : get_distance_photons(val)})
        dem_PE.update({gtx: get_DEM_from_photon_height(val)})

        if make_plots:

            if not os.path.isdir(os.path.join(base_dir, 'Exploration/figures/', data_fn, gtx)):
                os.mkdir(os.path.join(base_dir, 'Exploration/figures/', data_fn, gtx))

            max_distance = max(along_track_distance_PE[gtx])
            distance_step = 10000

            for distance_start in np.arange(0, max_distance, distance_step):
                nearest_start_index = utl.find_nearest(along_track_distance_PE[gtx], distance_start)
                nearest_end_index = utl.find_nearest(along_track_distance_PE[gtx], distance_start+distance_step)

                distance_slice = slice(nearest_start_index, nearest_end_index)

                utl.log('Figure photons - start {} / end {}'.format(distance_start, max(along_track_distance_PE[gtx])), log_level='INFO')

                plot_photon_height_single_beam(IS2_atl03_mds, gtx, along_track_distance_PE, index=distance_slice,
                                               coords=None, beams=None, outpath='F:/onderzoeken/thesis_msc/Exploration/figures/{}/{}/Photon_height_{}.png'.format(data_fn, gtx, distance_start))


    example_lakes = {'ID': [0, 1, 2, 3, 4, 5, 6],
                     'beam': ['gt2l', 'gt2l', 'gt2l', 'gt1r', 'gt1r', 'gt1l', 'gt3l'],
                     'lon': [(-49.05,-49.02), (-48.87, -48.84), (-48.38, -48.35), (-48.53, -48.50), (-48.45, -48.42), (-49.01, -48.98), (-48.79, -48.76)],
                     'lat': [(68.95, 69.04), (68.41, 68.50), (66.81, 66.90), (67.07, 67.17), (66.81, 66.90), (68.59, 68.68), (68.42, 68.51)]}


    plt.ioff()
    make_plots_examples = False
    for lake_ID in example_lakes['ID']:

        data_dict = get_classification_data(example_lakes, lake_ID, IS2_atl03_mds, along_track_distance_PE, dem_PE)

        if make_plots_examples:
            # -- create scatter plot of photon data for all beams
            outpath = os.path.join(base_dir, 'Exploration/figures/', 'test_{}_{}.png'.format(example_lakes['beam'][lake_ID], lake_ID))
            plot_lake(data_dict['distance'], data_dict['height'], data_dict['dem'], example_lakes, lake_ID, outpath)

        classification_df = pd.DataFrame(data_dict)
        classification_df.to_csv(os.path.join(base_dir, 'Exploration', 'testing_data_lake_{}.csv'.format(lake_ID)))

    print('finito')
