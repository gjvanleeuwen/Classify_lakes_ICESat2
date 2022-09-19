import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.raster_utils import gdal_calc

if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    data_dir = "F:/onderzoeken/thesis_msc/data/Sentinel/20190623"

    overwrite_NDWI = False
    s2_band_list = ['B03_10', 'B04_10', 'B08_10', 'B11_20'] # Green, Red, NIR, SWIR
    NDWI_calc = '(A - B) / (A + B)'

    s2_fn_list = pth.get_files_from_folder(data_dir, '*.SAFE')

    for subdir in s2_fn_list:

        s2_files = pth.get_sorted_s2_filelist(subdir, band_list=s2_band_list, recursive=True)

        NDWI_raster_dict = {'A': s2_files[0],
                            'B': s2_files[2]}
        NDWI_fn = s2_files[0][:-11] + "NDWI_10m.tif"
        utl.log(NDWI_fn, log_level='INFO')

        if not pth.check_existence(NDWI_fn, overwrite=overwrite_NDWI):
            gdal_calc(NDWI_raster_dict, NDWI_fn, NDWI_calc, dtype='Float32', no_data=-999,
                      create_options=("COMPRESS=LZW", "TILED=YES"),
                      file_format='GTiff', debug=True, overwrite=overwrite_NDWI)

        # ds = gdal.Warp(output_file, input_file, xRes=res, yRes=-res, format='GTIFF',
        #                creationOptions=['COMPRESS=LZW', 'TILED=YES'])









