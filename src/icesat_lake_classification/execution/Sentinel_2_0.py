import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.raster_utils import gdal_calc

from osgeo import gdal

if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    data_dir = "F:/onderzoeken/thesis_msc/data/Sentinel/20190620"

    calculate_NDWI = False
    calculate_NDWI_mask = True
    resample_SWIR = False

    overwrite_NDWI = False
    overwrite_NDWI_mask = True
    overwrite_SWIR = False

    s2_band_list = ['B03_10', 'B04_10', 'B08_10'] # Green, Red, NIR, SWIR
    NDWI_calc = '(A - B) / (A + B)'
    mask_calc = "1*logical_and(A<=1,A>=0.3)"

    s2_fn_list = pth.get_files_from_folder(data_dir, '*.SAFE')

    for subdir in s2_fn_list:

        s2_files = pth.get_sorted_s2_filelist(subdir, band_list=s2_band_list, recursive=True)

        NDWI_raster_dict = {'A': s2_files[0],
                            'B': s2_files[2]}
        NDWI_fn = s2_files[0][:-11] + "NDWI_10m.tif"

        if calculate_NDWI:

            if not pth.check_existence(NDWI_fn, overwrite=overwrite_NDWI):
                utl.log("Make NDWI " + NDWI_fn, log_level='INFO')
                gdal_calc(NDWI_raster_dict, NDWI_fn, NDWI_calc, dtype='Float32', no_data=-999,
                          create_options=("COMPRESS=LZW", "TILED=YES"),
                          file_format='GTiff', debug=True, overwrite=overwrite_NDWI)


        if calculate_NDWI_mask:

            NDWI_mask_fn = s2_files[0][:-11] + "NDWI_10m_mask.tif"
            NDWI_raster_dict2 = {'A': NDWI_fn}

            if not pth.check_existence(NDWI_mask_fn, overwrite=overwrite_NDWI_mask):
                utl.log("Make mask " + NDWI_mask_fn, log_level='INFO')
                gdal_calc(NDWI_raster_dict2, NDWI_mask_fn, mask_calc, dtype='Float32', no_data=-999,
                          create_options=("COMPRESS=LZW", "TILED=YES"),
                          file_format='GTiff', debug=True, overwrite=overwrite_NDWI_mask)


    if resample_SWIR:
        res = 10

        s2_fn_list = pth.get_files_from_folder(data_dir, '*.SAFE')

        for subdir in s2_fn_list:

            s2_files = pth.get_sorted_s2_filelist(subdir, band_list=['B03_10', 'B11_20'], recursive=True)
            resample_fn = s2_files[0][:-11] + "B11_10m.tif"

            if not pth.check_existence(resample_fn, overwrite=overwrite_SWIR):
                utl.log("resampling the SWIR band for : {}".format(s2_files[1]), log_level='INFO')

                ds = gdal.Warp(resample_fn, s2_files[1], xRes=res, yRes=-res, format='GTIFF',
                               creationOptions=['COMPRESS=LZW', 'TILED=YES'])
                ds = None








