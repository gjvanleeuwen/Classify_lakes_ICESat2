import icesat_lake_classification.utils as utl
from icesat_lake_classification.ICESat2_data_management import download_ICESat2_data

if __name__ == "__main__":
    __author__ = "gjvanleeuwen"
    __copyright__ = "gjvanleeuwen"
    __license__ = "MIT"
    uid = 'gijsvanleeuwen'
    email = 'g.j.vanleeuwens@gmail.com'

    utl.set_log_level(log_level='DEBUG')

    datapath = 'F:/onderzoeken/thesis_msc/data/ICESat2'

    ###     data track info:
    # bounding box
    product = 'ATL03'
    spatial_extent = [-50, 40, -40, 70]
    date_range = ['2019-03-01', '2019-08-31'] #, ['2020-03-01', '2020-08-31']]
    cycles = None
    orbitsegment = ['03', '05']
    tracks = ['727', '841', '1108', '1169']#, '1222']
    start_time = '00:00:00'
    end_time = '23:59:59'
    version = '005'

    for track in tracks:
        download_ICESat2_data(datapath, uid, email,
                                    spatial_extent, date_range, cycles, track,
                                    version='005', product='ATL03')

