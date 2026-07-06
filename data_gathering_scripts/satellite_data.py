import ee
import geemap
import numpy as np

ee.Authenticate()
ee.Initialize(project='air-pollution-501614')

def mask_clouds(img):
    scl = img.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask)

def get_patch(lat, lon, date_str, buffer_m=1280, days_window=15):
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m).bounds()
    date = ee.Date(date_str)
    coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(date.advance(-days_window, 'day'), date.advance(days_window, 'day'))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .map(mask_clouds))
    image = coll.median().select(['B4', 'B3', 'B2', 'B8'])
    return image.clip(region)

def get_patch_array(lat, lon, date_str, buffer_m=1280, scale=10):
    image = get_patch(lat, lon, date_str, buffer_m=buffer_m)
    region = ee.Geometry.Point([lon, lat]).buffer(buffer_m).bounds()
    arr = geemap.ee_to_numpy(image, region=region, scale=scale)
    return arr

if __name__ == '__main__':
    patch = get_patch_array(lat=48.7758, lon=9.1829, date_str='2023-06-15') #should be stuttgart
    print(patch.shape)