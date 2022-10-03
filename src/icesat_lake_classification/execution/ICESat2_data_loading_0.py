import os
import pandas as pd

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.ICESat2_data_management import load_ICESat2_ATL03_data, get_cum_along_track_distance, get_dem_height_beam_ph

if __name__ == "__main__":
    utl.set_log_level(log_level='INFO')

    ### Tasks
    overwrite_data = True

    data_dir = 'F:/onderzoeken/thesis_msc/data/ICESat2'
    outdir = 'F:/onderzoeken/thesis_msc/data/ICESat2_csv'

    track_fn_list = pth.get_files_from_folder(data_dir, '*ATL03*1222*.h5')
    utl.log(track_fn_list, log_level='INFO')

    for fn in track_fn_list:
        utl.log('1. -- load ICESat ATL03 track {} - Create Classification dataset'.format(fn), log_level='INFO')
        file_info, track_data, track_attributes, beam_names = load_ICESat2_ATL03_data(fn)

        utl.log('2. -- retrieving along track distance for all photons in track {}'.format(fn), log_level='INFO')
        along_track_distance_dict = get_cum_along_track_distance(track_data)

        utl.log('3. -- retrieving along track distance for all photons in track {}'.format(fn), log_level='INFO')
        dem_h_dict = get_dem_height_beam_ph(track_data)

        utl.log('3. -- saving photon and geo data per beam for classification', log_level='INFO')

        data_dict = {}
        rgt = file_info['rgt']
        date = str(file_info['datetime'].date())

        for beam in ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']:
            n_ph = len(track_data[beam]['heights']['h_ph'])

            if not pth.check_existence(
                    os.path.join(outdir, 'ATL03_photon_data_track_{}_date_{}_beam_{}_nph_{}.csv'.format(
                            rgt, date, beam, n_ph)), overwrite=overwrite_data):

                utl.log('4a. -- Processing beam {} for RGT {} and date {}'.format(beam, rgt, date), log_level='INFO')

                beam_track_data = track_data[beam]
                beam_along_track_distance = along_track_distance_dict[beam]
                beam_demh = dem_h_dict[beam]

                height_ph = beam_track_data['heights']['h_ph']
                lon_ph = beam_track_data['heights']['lon_ph']
                lat_ph = beam_track_data['heights']['lat_ph']

                data_dict_temp = {'height': height_ph,
                                  'distance': beam_along_track_distance,
                                  'dem': beam_demh,
                                  'lon': lon_ph, 'lat': lat_ph
                                  }

                classification_df = pd.DataFrame(data_dict_temp)

                if outdir:
                    classification_df.to_csv(os.path.join(outdir,
                                                          'ATL03_photon_data_track_{}_date_{}_beam_{}_nph_{}.csv'.format(
                                                              rgt, date, beam, n_ph)))

                data_dict.update({beam: classification_df})
            else:
                utl.log(
                    '4a. -- beam {} for RGT {} and date {} - Already exists, not processing'.format(beam, rgt, date),
                    log_level='INFO')
                data_dict.update({beam: {}})


