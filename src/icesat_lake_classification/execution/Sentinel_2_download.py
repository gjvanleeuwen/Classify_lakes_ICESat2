# connect to the API
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import tenacity
import icesat_lake_classification.utils as utl

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(3600))
def download_all(products, outpath):
    return api.download_all(products, directory_path=outpath)



if __name__ == "__main__":
    utl.log("start api login", log_level="INFO")
    api = SentinelAPI('ertyboi', 'Sentinel1!', 'https://scihub.copernicus.eu/dhus')

    outpath = "F:/onderzoeken/thesis_msc/data/Sentinel/20190617"
    geometry = {"type":"Polygon","coordinates":[[[-49.509888,69.597805],[-47.488403,69.553715],[-47.471924,67.32716],[-48.076172,64.031339],[-49.559326,63.821288],[-49.509888,69.597805]]]}

    # search by polygon, time, and SciHub query keywords
    footprint = geojson_to_wkt(geometry)
    utl.log("query products", log_level="INFO")
    products = api.query(footprint,
                         date=('20190616', '20190618'),
                         platformname='Sentinel-2',
                         cloudcoverpercentage=(0, 15),
                         producttype='S2MSI2A')

    downloaded, triggered, failed = download_all(products,outpath=outpath)

    utl.log(downloaded, log_level='INFO')
    utl.log(triggered, log_level='INFO')
    utl.log(failed, log_level='INFO')

