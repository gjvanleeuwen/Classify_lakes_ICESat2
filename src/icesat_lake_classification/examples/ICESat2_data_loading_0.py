import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.ICESat2_data_management import load_ICESat2_ATL03_data, get_classification_data, get_cum_along_track_distance

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

    track_fn_list = pth.get_files_from_folder(data_dir, '*ATL03*12221*.h5')
    utl.log(track_fn_list, log_level='INFO')

    for fn in track_fn_list:
        utl.log('1. -- load ICESat ATL03 track {} - Create Classification dataset'.format(fn), log_level='INFO')
        file_info, track_data, track_attributes, beam_names = load_ICESat2_ATL03_data(fn)

        utl.log('2. -- retrieving along track distance for all photons in track {}'.format(fn), log_level='INFO')
        along_track_distance_dict = get_cum_along_track_distance(track_data)

        if save_data:
            utl.log('3. -- saving photon and geo data per beam for classification', log_level='INFO')
            classification_data = get_classification_data(track_data, along_track_distance_dict, file_info, outdir=classification_dir, overwrite=True) #classification_dir)

