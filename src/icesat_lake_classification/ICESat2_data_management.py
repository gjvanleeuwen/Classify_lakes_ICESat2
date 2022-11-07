import numpy as np
import pandas as pd
import scipy

from icesat2_toolkit.read_ICESat2_ATL03 import read_HDF5_ATL03

import icesat_lake_classification.validation as validation
import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth



def download_ICESat2_data(outpath, earthdata_uid, email,
                                spatial_extent, date_range, cycles, tracks,
                                start_time='00:00:00', end_time='23:59:59', version='005', product='ATL03', download=True):
    """
    Function to query and/or download all ICESat-2 data files in a certain region. time range and product + version

    Parameters
    ----------
    outpath : str
        base directory in which to save the downloaded ICESat-2 files
    earthdata_uid : str
        Username for the NASA earthdata website
    email : str
        email of your NASA earthdata account.
    spatial_extent : list,str, list of lists
        a region of interest to search within. This can be entered as a bounding box, polygon vertex coordinate pairs,
        or a polygon geospatial file (currently shp, kml, and gpkg are supported).

        bounding box: Given in decimal degrees for the lower left longitude, lower left latitude, upper right longitude, and upper right latitude
        polygon vertices: Given as longitude, latitude coordinate pairs of decimal degrees with the last entry a repeat of the first.
        polygon file: A string containing the full file path and name.
    date_range :list
        the date range for which you would like to search for results. Must be formatted as a set of 'YYYY-MM-DD' strings separated by a comma.
    cycles : str, list of strings
        Which orbital cycle to use, input as a numerical string or a list of strings. If no input is given,
        this value defaults to all available cycles within the search parameters. An orbital cycle refers to the 91-day repeat period of the ICESat-2 orbit.
    tracks : str, list of strings
        Which Reference Ground Track (RGT) to use, input as a numerical string or a list of strings.
        If no input is given, this value defaults to all available RGTs within the spatial and temporal search parameters.
    start_time
    end_time
    version : str, optional
        String indicating the verson of data to download, older versions may not be accesible anymore when new ones get released.
        check  https://nsidc.org/data/icesat-2/data-sets for the latest version.
    product : str, optional
        the data product of interest, known as its "short name". See https://nsidc.org/data/icesat-2/data-sets for a list of the available data products.
    download : bool, optional
        if True queried data will be downloaded, otherwise only the query is done and id list is returned.

    Returns
    -------
    iterable of strings with all names of downloaded files.
    """
    import icepyx as ipx
    region = ipx.Query(product=product, spatial_extent=spatial_extent,
                       date_range=date_range, start_time=start_time, end_time=end_time,
                       cycles=cycles, tracks=tracks, version=version)
    region.earthdata_login(earthdata_uid, email)

    utl.log(region.avail_granules(ids=True), log_level='INFO')
    utl.log(region.product_summary_info(), log_level='INFO')

    region.order_granules()
    if download:
        region.download_granules(outpath)

    return region.avail_granules(ids=True)



def load_ICESat2_ATL03_data(path):
    """
    Reads ICESat-2 ATL03 Global Geolocated Photons data files

    Parameters
    ----------
    path : str
        full path to an IceSat-2 ATL03 file

    Returns
    -------
    file_info : dict
        Dictionary with information parsed from the filename like: track, date, version etc
    data : dict
        dictionary with ATL03 variables datasets sorted per beam
    attributes: dict
        dictionary with ATL03 attributes data sorted per beam
    beam_names: list
        list of strings with valid ICESat-2 beams within ATL03 file

    """
    utl.log('Loading datafile"{}'.format(pth.get_filename(path)), log_level='INFO')

    if not pth.check_existence(path):
        utl.log('file to open does not exist', log_level='ERROR')
        pass

    file_info = pth.parse_ATL03_fullpath(path)
    data, attributes, beam_names = read_HDF5_ATL03(path, ATTRIBUTES=True)

    utl.log('Completed loading of file', log_level='INFO')

    return file_info, data, attributes, beam_names


def get_geo_info_beam(beam_data, outpath=None):
    """
    Function to get the reference coordinates of a specific beam in an ATL03 dataset

    Parameters
    ----------
    beam_data : dict
        Dictionary from Load_icesat data with the data for 1 beam of an ATL03 track
    outpath : str, optional
        will write lon,lat to a text file if outpath is given

    Returns
    -------
    n_seg : int
        total number of segments in the beam of this track
    n_ph : int
        total number of photons in the beam of this track
    coordinates: nd.array
        2 dimensional array with the matched lon/lat of each segments reference photon

    """
    n_seg = len(beam_data['geolocation']['segment_id'])
    n_ph = beam_data['heights']['delta_time'].shape[0]

    coordinates = np.vstack((np.array(beam_data['geolocation']['reference_photon_lon']),
                             np.array(beam_data['geolocation']['reference_photon_lat']))).T

    if outpath:
        pd.DataFrame(coordinates).to_csv(outpath)

    return n_seg, n_ph, coordinates


def get_cum_along_track_distance(track_data, beams=None):
    """
    Function to retrieve the full along track distance for each single photon in an ATL03 track

    Parameters
    ----------
    track_data : dict
        dictionary with ATL03 data dictionaries for each beam listed in "beams"
    beams : list, optional
        list of strings with each string being a name of an ATL03 track beam ex ["gt1r", "gt2l"]
        if none all beams in the Track_data will be processed.

    Returns
    -------
    along_track_distance_dict : dict
        dictionary with beams as keys and cumalitive along track distance for each photon as an array as value

    """
    along_track_distance_dict = {}
    if not beams:
        beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']

    for beam in beams:

        n_ph_segment = track_data[beam]['geolocation']['segment_ph_cnt']  # -- number of photon events in the segment
        distance_segment = track_data[beam]['geolocation']['segment_length']   # -- along-track distance for each ATL03 segment
        along_distance_ph_segment = track_data[beam]['heights']['dist_ph_along']  # along track distance for each photon from start of segment

        cumalative_segment_distance = np.cumsum(np.insert(distance_segment,0,0))[:-1]
        cumalative_segment_distance_ph = [segment_length for segment_index, segment_length in
                                        enumerate(cumalative_segment_distance) for i in
                                        range(n_ph_segment[segment_index])]

        along_distance_ph = cumalative_segment_distance_ph + along_distance_ph_segment
        along_track_distance_dict.update({beam : along_distance_ph})

    return along_track_distance_dict

def get_dem_height_beam_ph(track_data, beams=None):

    dem_height_dict = {}
    if not beams:
        beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']

    for beam in beams:

        n_pe = track_data[beam]['heights']['delta_time'].shape
        # -- along-track and across-track distance for photon events
        x_atc = track_data[beam]['heights']['dist_ph_along']
        # -- digital elevation model interpolated to photon events
        dem_h = np.zeros((n_pe))

        # -- first photon in the segment (convert to 0-based indexing)
        Segment_Index_begin = track_data[beam]['geolocation']['ph_index_beg'] - 1
        # -- number of photon events in the segment
        Segment_PE_count = track_data[beam]['geolocation']['segment_ph_cnt']
        # -- along-track distance for each ATL03 segment
        Segment_Distance = track_data[beam]['geolocation']['segment_dist_x']


        # -- for each 20m segment
        for j, _ in enumerate(track_data[beam]['geolocation']['segment_id']):
            # -- index for 20m segment j
            idx = Segment_Index_begin[j]
            # -- skip segments with no photon events
            if (idx < 0):
                continue
            # -- number of photons in 20m segment
            cnt = Segment_PE_count[j]
            # -- add segment distance to along-track coordinates
            x_atc[idx:idx + cnt] += Segment_Distance[j]
            # -- interpolate digital elevation model to photon events
            dem_h[idx:idx + cnt] = track_data[beam]['geophys_corr']['dem_h'][j]

        dem_height_dict.update({beam: dem_h})

    return dem_height_dict


def get_ph_index_from_coordinates(longitude_data, min_lon, max_lon):

    min_lon_index, max_lon_index = utl.find_nearest(longitude_data, min_lon), utl.find_nearest(
        longitude_data, max_lon)

    return slice(min_lon_index, max_lon_index)


def get_classification_data_for_lake_from_disk(data_dir, beam, rgt, min_lon, max_lon):

    data_path = pth.get_files_from_folder(data_dir, '*{}*{}*.csv'.format(rgt,beam))
    lake_data = pd.read_csv(data_path)

    index_slice = get_ph_index_from_coordinates(lake_data['lon'], min_lon, max_lon)

    return lake_data.iloc[:,index_slice]


def get_background_rate (val):
    # -- Transmit time of the reference photon
    delta_time = val['geolocation']['delta_time']
    # -- interpolate background photon rate based on 50-shot summation
    background_delta_time = val['bckgrd_atlas']['delta_time']
    SPL = scipy.interpolate.UnivariateSpline(background_delta_time,
                                             val['bckgrd_atlas']['bckgrd_rate'], k=3, s=0)
    background_rate = SPL(delta_time)

    return background_rate


# def add_rows_to_df(input_rows):
#     rows_list = []
#     for row in input_rows:
#         dict1 = {}
#         # get input row in dictionary format
#         # key = col_name
#         dict1.update(blah..)
#
#         rows_list.append(dict1)
#
#     df = pd.DataFrame(rows_list)


def add_surface_line(classification_df, window_size, sample_rate):
    surface_df = classification_df.loc[classification_df['clusters'] == 2]
    surface_df.sort_values('distance', inplace=True)
    surface_line = pd.DataFrame(
        utl.rollBy(surface_df['height'], surface_df['distance'], window_size, sample_rate,
                   np.nanmedian))
    result_index = [utl.find_nearest_sorted(surface_df['distance'].values, value) for value in surface_line.index]

    surface_line['idx'] = surface_df['distance'].iloc[result_index].index
    surface_line = surface_line.reset_index().set_index('idx')
    surface_line.rename(columns={'index': 'distance', 0: 'height'}, inplace=True)
    surface_line['distance'] = surface_line['distance'] + (window_size / 2)

    surface_df_window = utl.interpolate_df_to_new_index(surface_df,
                                                        surface_line.loc[:, ['distance', 'height']].copy(), 'distance')

    df_interp = utl.interpolate_df_to_new_index(classification_df,
                                                surface_df_window.loc[:, ['distance', 'height']].copy(),
                                                'distance')

    classification_df['surface_height'] = df_interp['height'].copy()
    classification_df['surface_distance'] = df_interp['distance'].copy()

    return classification_df


def add_bottom_line(classification_df, window_size, sample_rate, buffer_bottom_line, refractive_index):
    # calculate slope
    classification_df['slope'] = np.abs((np.roll(classification_df['surface_height'], 1) - classification_df['surface_height']) / (
                np.roll(classification_df['surface_distance'], 1) - classification_df['surface_distance']))  # abs(rise/run)

    # calculate distance from surface for every photon
    classification_df['ph_depth'] = utl.perpendicular_distance(classification_df['surface_height'], classification_df['height'],
                                                       classification_df['slope'])

    # adjust height of bottom photons and change clusters further - calculate difference between surface and bottom photons
    bottom_index = np.where((classification_df['clusters'] == 3) & (classification_df['ph_depth'] < buffer_bottom_line))
    # real height = surface - current_depth/1.33
    classification_df['height'].iloc[bottom_index] = classification_df['surface_height'].iloc[bottom_index] - \
                                                     (np.abs(
                                                         classification_df['ph_depth'].iloc[bottom_index]) / refractive_index)
    classification_df['clusters'].iloc[
        np.where((classification_df['clusters'] == 3) & (classification_df['ph_depth'] < buffer_bottom_line))] = 5

    # seperate bottom photons
    bottom_df = classification_df.iloc[bottom_index].copy()

    bottom_df.sort_values('distance', inplace=True)
    bottom_line = pd.DataFrame(
        utl.rollBy(bottom_df['height'], bottom_df['distance'], window_size, sample_rate,
                   np.nanmedian))
    result_index = [utl.find_nearest_sorted(bottom_df['distance'].values, value - (window_size)) for value in
                    bottom_line.index]

    bottom_line['idx'] = bottom_df['distance'].iloc[result_index].index
    bottom_line = bottom_line.reset_index().set_index('idx')
    bottom_line.rename(columns={'index': 'distance', 0: 'height'}, inplace=True)
    bottom_line['distance'] = bottom_line['distance'] + (window_size / 2)

    bottom_df_window = utl.interpolate_df_to_new_index(bottom_df,
                                                       bottom_line.loc[:, ['distance', 'height']].copy(), 'distance')

    bottom_df_interp = utl.interpolate_df_to_new_index(classification_df,
                                                       bottom_df_window.loc[:, ['distance', 'height']].copy(),
                                                       'distance')

    classification_df['bottom_distance'] = bottom_df_interp['distance'].copy()
    classification_df['bottom_height'] = bottom_df_interp['height'].copy()

    return classification_df


def calc_ph_statistics(classification_df, window_size=10000):
    classification_df['SurfBottR'], classification_df['SurfNoiseR'], \
    classification_df['range'], classification_df['slope_mean'], classification_df['dem_diff'] = 0, 0, 0, 0, 0

    # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, 5 is real bottom
    for i, (ph_start, ph_end) in enumerate(zip(np.arange(0, len(classification_df), window_size),
                                               np.append(np.arange(window_size, len(classification_df), window_size),
                                                         len(classification_df) - 1))):
        index_slice = slice(ph_start, ph_end)

        n_noise_ph = classification_df['clusters'].iloc[index_slice].value_counts()[0]  # 0 = noise
        n_signal_ph = len(classification_df) - n_noise_ph
        if 5 in classification_df['clusters'].iloc[index_slice].value_counts():
            n_bottom_ph = classification_df['clusters'].iloc[index_slice].value_counts()[5]
        else:
            n_bottom_ph = 1  # bottom
        if 2 in classification_df['clusters'].iloc[index_slice].value_counts():
            n_surface_ph = classification_df['clusters'].iloc[index_slice].value_counts()[2]
        else:
            n_surface_ph = 1  # surface

        classification_df['SurfBottR'].iloc[index_slice] = n_surface_ph / n_bottom_ph  # around 5...
        classification_df['SurfNoiseR'].iloc[index_slice] = n_surface_ph / n_noise_ph  # around 5...
        classification_df['range'].iloc[index_slice] = np.abs(
            classification_df['height'].iloc[index_slice].max() - classification_df['height'].iloc[
                index_slice].min())
        classification_df['dem_diff'].iloc[index_slice] = np.max(
            np.abs(classification_df['dem'].iloc[index_slice] - classification_df['height'].iloc[
                index_slice]))
        classification_df['slope_mean'].iloc[index_slice] = classification_df['slope'].iloc[
            index_slice].mean()

    return classification_df


def classify_lake(classification_df, lake_boundary, slope_boundary, SNR_bound=0.5, SBR_bound=100, SBR_bound2=0.5):
    classification_df['lake_diff'] = utl.perpendicular_distance(classification_df['surface_height'],
                                                                classification_df['bottom_height'],
                                                                classification_df['slope'])

    classification_df['lake'] = np.where(
        (classification_df['lake_diff'] < lake_boundary) & (np.abs(classification_df['slope']) < slope_boundary)
        & (classification_df['SurfNoiseR'] > SNR_bound) & (classification_df['SurfBottR'] < SBR_bound)
        & (classification_df['SurfBottR'] > SBR_bound2), 1, 0)
    classification_df['lake'].iloc[np.where(
        (classification_df['lake_diff'] < lake_boundary) & (np.abs(classification_df['slope']) >= slope_boundary)
        & (classification_df['SurfNoiseR'] > SNR_bound) & (classification_df['SurfBottR'] < SBR_bound)
        & (classification_df['SurfBottR'] > SBR_bound2))] = 2
    classification_df['lake'].iloc[np.where(
        (np.abs(classification_df['slope']) < slope_boundary) & (classification_df['lake_diff'] >= lake_boundary)
        & (classification_df['SurfNoiseR'] > SNR_bound) & (classification_df['SurfBottR'] < SBR_bound)
        & (classification_df['SurfBottR'] > SBR_bound2))] = 3

    return classification_df


