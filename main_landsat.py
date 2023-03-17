# import numpy
# from pyhdf import HDF
from pyhdf.SD import SD, SDC
from osgeo import gdal
from osgeo import ogr
# from glob import glob
# import geopandas as gpd
# import xarray as xr
# import rioxarray as rxr
# from osgeo import osr
# from osgeo import gdal_array
# from osgeo import gdalconst
# import h5py
import os
# import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
# from eoreader.reader import Reader
# from eoreader.bands import *
# import rasterio
from time import time
# from sentinelhub import geo_utils, BBox, bbox_to_dimensions, CRS
import xarray
import cv2
from PIL import Image

def save_as_tiff(massive, way, name):
    Image.fromarray(massive).save(way+name+".tif")
# получение координат для sentinel, гордость моего кота(юриного)
def convert_to_deciminal_degree(degree):
    a = float(degree[0:degree.find('d')])
    b = float(degree[degree.find('d') + 1:degree.find('\'')])
    c = float(degree[degree.find('\'') + 1:degree.find('"')])
    return (a + (b / 60) + (c / 3600))


from pyproj import Proj, transform


def sentinel_coordinates(way):
    import warnings
    warnings.filterwarnings('ignore')
    way = get_ways_sentinel(way)['B08']
    rastr = xarray.open_dataset(way)
    info = gdal.Info(way)
    cordsys = info[info.find(',', info.find('ID["EPSG"', info.find('ORDER[2]'))) + 1:info.find(']]',
                                                                                               info.find('ID["EPSG"',
                                                                                                         info.find(
                                                                                                             'ORDER[2]')))]
    lat = np.array(rastr['x'])
    lon = np.array(rastr['y'])
    from pyproj import Proj, transform
    lonlat = Proj(init="epsg:" + str(cordsys))
    sphmerc = Proj(init="epsg:4326")
    ll = (lat, lon)
    sm = np.array(transform(lonlat, sphmerc, *ll))
    return {'lat': sm[0], 'lon': sm[1]}


# def sentinel_coordinates(way):
#     sentinel_coordinates1(way)
#     info = gdal.Info(get_ways_sentinel(way)['B08'])
#     # print(info)
#     Up_Left = info[
#               info.find("(", info.find("(", info.find("Upper Left")) + 1):info.find("\n",
#                                                                                     info.find("Upper Left"))].replace(
#         '(', '').replace(')', '').replace(' ', '').replace('E', '').replace('N', '').split(',')
#     Lower_Right = info[info.find("(", info.find("(", info.find("Lower Right")) + 1):info.find("\n", info.find(
#         "Lower Right"))].replace('(', '').replace(')', '').replace(' ', '').replace('E', '').replace('N', '').split(',')
#     coords = [convert_to_deciminal_degree(Up_Left[0]), convert_to_deciminal_degree(Up_Left[1]),
#               convert_to_deciminal_degree(Lower_Right[0]), convert_to_deciminal_degree(Lower_Right[1])]
#     resolution = 40
#     loc_bbox = BBox(bbox=coords, crs=CRS.WGS84)
#     loc_size = bbox_to_dimensions(loc_bbox, resolution=resolution)
#     bb_utm = geo_utils.to_utm_bbox(loc_bbox)
#     transf = bb_utm.get_transform_vector(resx=resolution, resy=resolution)
#     pix_lat = np.array(np.arange(0, loc_size[1]))
#     lats = np.array([pix_lat] * loc_size[0]).transpose()
#     pix_lon = np.array(np.arange(0, loc_size[0]))
#     lons = np.array([pix_lon] * loc_size[1])
#     lon, lat = geo_utils.pixel_to_utm(lats, lons, transf)
#     lon_degrees, lat_degrees = geo_utils.to_wgs84(lon, lat, bb_utm.crs)
#     return {'lat': np.array(lat_degrees), 'lon': np.array(lon_degrees)}

# воимя отца сына и святого дуба помоги этому коду заработаать


# Блок функций для открытия снимков
def get_names_landsat(way):
    a = os.listdir(way)
    channals = {}
    for el in a:
        if el.endswith(".TIF") and el[-5] in "0123456789":
            channals[f"B{el[-5]}"] = way + "/" + el
    return channals


def get_names_sentinel(way, band):
    prod = Reader().open(way)
    mas = prod.load([band])
    return mas[list(mas.keys())[0]][0]


def get_ways_sentinel(way):
    ways_slov = dict()
    for root, dirs, files in os.walk(way):
        # print(files)
        for filenames in files:
            if filenames[0] == "T" and (filenames[-7:-4] == "20m"):
                if filenames[-11:-8] in ["B05", "B06", "B07", "B8A", "B11", "B12"]:
                    ways_slov[filenames[-11:-8]] = root + "/" + filenames
                    # print(root + "/" + filenames)
            elif filenames[0] == "T" and (filenames[-7:-4] == "60m"):
                if filenames[-11:-8] in ["B01", "B09", 'B10']:
                    ways_slov[filenames[-11:-8]] = root + "/" + filenames
            elif filenames[0] == "T" and (filenames[-7:-4] == "10m"):
                if filenames[-11:-8] in ["B02", "B03", "B04", "B08"]:
                    ways_slov[filenames[-11:-8]] = root + "/" + filenames
            elif filenames[0] == "T":
                ways_slov[filenames[-7:-4]] = root + "/" + filenames
            # print(filenames[-7:-4], filenames[0])
            # ways_slov[filenames[-7:-4]] = root + "/" + filenames
    return ways_slov


# этот кал надо переделать, он медленный и через eoreader
def get_cordinates_sentinel(file, x, y):
    return [float(file[x][y].coords['x']), float(file[x][y].coords['y'])]


# я хуй его знает для чего это надо
def LatLon_from_XY(ProductSceneGeoCoding, x, y):
    # From x,y position in satellite image (SAR), get the Latitude and Longitude
    geopos = ProductSceneGeoCoding.getGeoPos((x, y), None)
    latitude = geopos.getLat()
    longitude = geopos.getLon()
    return latitude, longitude


# Блок функций расчета индексов для landsat 8
def ndvi(way):
    np.seterr(divide='ignore', invalid='ignore')
    channals = get_names_landsat(way)
    NIR = gdal.Open(channals["B4"]).ReadAsArray().astype("float32")
    RED = gdal.Open(channals["B3"]).ReadAsArray()
    ndvi_ = (NIR - RED) / (NIR + RED)
    plt.imshow(ndvi_)
    plt.show()
    return ndvi_


def ndsi(way):
    np.setter(divide='ignore', invalid='ignore')
    channels = get_names_landsat(way)
    green = gdal.Open(channels['B2']).ReadAsArray().astype('float32')
    swir = gdal.Open(channels['B5']).ReadAsArray().astype('float32')
    ndsi_ = (green - swir) / (green + swir)
    plt.imshow(ndsi_)
    plt.show()
    return ndsi_


def ndfsi(way):
    np.setter(divide='ignore', invalid='ignore')
    channels = get_names_landsat(way)
    nir = gdal.Open(channels['B4']).ReadAsArray().astype('float32')
    swir = gdal.Open(channels['B5']).ReadAsArray().astyper('float32')
    ndfsi_ = (nir - swir) / (nir + swir)
    plt.imshow(ndfsi_)
    plt.show()
    return ndfsi_


def mndwi(way):
    np.setter(divide='ignore', invalid='ignore')
    channels = get_names_landsat(way)
    green = gdal.Open(channels['B2']).ReadAsArray().astype('float32')
    swir = gdal.Open(channels['B7']).ReadAsArray().astype('float32')
    mndwi_ = (green - swir) / (green + swir)
    plt.imshow(mndwi_)
    plt.show()
    return mndwi_


# Блок функций расчета индексов для sentinel 2
def sentinel_ndvi(way):
    nir = gdal.Open(get_ways_sentinel(way)['B08']).ReadAsArray()
    red = gdal.Open(get_ways_sentinel(way)['B04']).ReadAsArray()
    ndvi_sentinel = (nir - red) / (nir + red)
    plt.imshow(ndvi_sentinel)
    plt.show()
    return ndvi_sentinel


def sentinel_ndsi(way):
    b3 = gdal.Open(get_ways_sentinel(way)['B03']).ReadAsArray()
    b12 = gdal.Open(get_ways_sentinel(way)['B11']).ReadAsArray()
    swir = np.repeat(b12, 2, axis=1).astype('float32')
    swir = np.repeat(swir, 2, axis=0).astype('float32')
    ndsi_sentinel = (b3 - swir) / (b3 + swir)
    plt.imshow(ndsi_sentinel)
    plt.show()
    return ndsi_sentinel


import json


def sentinel_mndwi(way):
    b3 = gdal.Open(get_ways_sentinel(way)['B03']).ReadAsArray().astype('float32')
    b12 = gdal.Open(get_ways_sentinel(way)['B11']).ReadAsArray()
    swir = np.repeat(b12, 2, axis=1).astype('float32')
    swir = np.repeat(swir, 2, axis=0).astype('float32')
    mndwi_sentinel = (b3 - swir) / (b3 + swir)
    mndwi_sentinel[np.logical_and((swir == 0), (b3 == 0))] = 0
    mndwi_sentinel[mndwi_sentinel<0]=None
    mndwi_sentinel[mndwi_sentinel>=0]=1
    # plt.imshow(mndwi_sentinel)
    # plt.show()
    # "C:\Users\perminov_u\Desktop\image1.txt.txt"
    # print(np.count_nonzero(mndwi_sentinel>0)*100/1000000, 'square killometrs')
    return mndwi_sentinel


def sentinel_ndgr(way):
    b3 = gdal.Open(get_ways_sentinel(way)['B03']).ReadAsArray().astype('float32')
    b4 = gdal.Open(get_ways_sentinel(way)['B04']).ReadAsArray().astype('float32')
    ndgr_sentinel = (b3 - b4) / (b3 + b4)
    # ndgr_sentinel[ndgr_sentinel<0]=0
    # ndgr_sentinel[ndgr_sentinel>0]=1
    plt.imshow(ndgr_sentinel)
    plt.show()
    return ndgr_sentinel


def sentinel_ndfsi(way):
    b12 = gdal.Open(get_ways_sentinel(way)['B11']).ReadAsArray()
    swir = np.repeat(b12, 2, axis=1).astype('float32')
    swir = np.repeat(swir, 2, axis=0).astype('float32')
    nir = gdal.Open(get_ways_sentinel(way)['B08']).ReadAsArray()
    ndfsi_sentinel = (nir - swir) / (nir + swir)
    plt.imshow(ndfsi_sentinel)
    plt.show()
    return ndfsi_sentinel


# import sentinelhub


# функция для облачков

# разница водного игдекса, но при желании можно перековырять под другой
def sentinel_corner_coordinates1(way):
    info = gdal.Info(get_ways_sentinel(way)['B08'])
    corners = ['Upper Left', 'Upper Right', 'Lower Left']
    corns_cords = []
    for i in corners:
        temp = info[info.find("(", info.find("(", info.find(i)) + 1):info.find("\n", info.find(i))].replace('(',
                                                                                                            '').replace(
            ')', '').replace(' ', '').replace('E', '').replace('N', '').split(',')
        corns_cords.append([temp[0], temp[1]])
    for i in range(3):
        corns_cords[i] = [convert_to_deciminal_degree(corns_cords[i][0]),
                          convert_to_deciminal_degree(corns_cords[i][1])]
    return corns_cords
def sentinel_corner_coordinates(cords):
    # info = gdal.Info(get_ways_sentinel(way)['B08'])
    # corners = ['Upper Left', 'Upper Right', 'Lower Left']
    # corns_cords = []
    # for i in corners:
    #     temp = info[info.find("(", info.find("(", info.find(i)) + 1):info.find("\n", info.find(i))].replace('(',
    #                                                                                                         '').replace(
    #         ')', '').replace(' ', '').replace('E', '').replace('N', '').split(',')
    #     corns_cords.append([temp[0], temp[1]])
    # for i in range(3):
    #     corns_cords[i] = [convert_to_deciminal_degree(corns_cords[i][0]),
    #                       convert_to_deciminal_degree(corns_cords[i][1])]
    # return corns_cords
    return [[cords['lat'][0], cords['lon'][0]], [cords['lat'][-1], cords['lon'][0]], [cords['lat'][0], cords['lon'][-1]]]


def get_centeres(way):
    info = gdal.Info(get_ways_sentinel(way)['B08'])
    print(info)
    temp = info[info.find("(", info.find("(", info.find('Center')) + 1):info.find("\n", info.find('Center'))].replace(
        '(', '').replace(')', '').replace(' ', '').replace('E', '').replace('N', '').split(',')
    temp = {'lat': convert_to_deciminal_degree(temp[0]), 'lon': convert_to_deciminal_degree(temp[1])}
    return temp


def closer_value_search(mas, value):
    coef = 1
    min = np.min(mas)
    max = np.max(mas)
    anw = 0
    if mas[0] < mas[-1]:
        if value < min:
            value = min + (min - value)
            coef = -1
        if value > max:
            anw = mas.shape[0]
            value = min + (value - max)
    if mas[0] > mas[-1]:
        if value < min:
            anw = mas.shape[0]
            value = max - (min - value)
        if value > max:
            coef = -1
            value = max - (value - max)
    tmp = np.abs(mas - value)
    anw += np.where(tmp == tmp.min())[0][0]
    return anw * coef

def whater_difference(way, way1):
    center = get_centeres(way)
    center1 = get_centeres(way1)
    cords = sentinel_coordinates(way)
    if center==center1:
        corns1 = sentinel_corner_coordinates(cords)
    else:
        corns1=sentinel_corner_coordinates1(way)
    points = []
    for i in range(3):
        points.append([closer_value_search(cords['lat'], corns1[i][0] + (center['lat'] - center1['lat'])),
                       closer_value_search(cords['lon'], corns1[i][1] + (center['lon'] - center1['lon']))])
    rastr = sentinel_mndwi(way)
    rastr11 = sentinel_mndwi(way1)
    rows, cols = rastr.shape[:2]
    transform_matrix = cv2.getAffineTransform(np.float32([[0, 0], [rows - 1, 0], [0, cols - 1]]), np.float32(points))
    rastr1 = cv2.warpAffine(rastr, transform_matrix, (cols, rows))
    c=np.zeros((rastr11.shape[0], rastr11.shape[1]))
    c[np.logical_and((rastr1!=0), (rastr1!=1))]=1
    # (np.logical_and((rastr11 != 0), (rastr11 != 1)), (rastr1 == 1)),
    np.logical_or(np.logical_and(np.logical_and((rastr1!=0), (rastr1!=1)), (rastr11==1)), np.logical_and(np.logical_and((rastr11!=0), (rastr11!=1)), (rastr1==1)), out=c)
    plt.imshow(c)
    plt.show()
    return c


# def whater_difference(way, way1):
#     center = get_centeres(way)
#     center1 = get_centeres(way1)
#     cords = sentinel_coordinates(way)
#     if center==center1:
#         corns1 = sentinel_corner_coordinates(cords)
#     else:
#         corns1=sentinel_corner_coordinates1(way)
#     points = []
#     for i in range(3):
#         points.append([closer_value_search(cords['lat'], corns1[i][0] + (center['lat'] - center1['lat'])),
#                        closer_value_search(cords['lon'], corns1[i][1] + (center['lon'] - center1['lon']))])
#     print(cords)
#     print(corns1)
#     print(points)
#     rastr = gdal.Open(get_ways_sentinel(way)['B08']).ReadAsArray()
#     rastr11 = gdal.Open(get_ways_sentinel(way1)['B08']).ReadAsArray()
#     rows, cols = rastr.shape[:2]
#     transform_matrix = cv2.getAffineTransform(np.float32([[0, 0], [rows - 1, 0], [0, cols - 1]]), np.float32(points))
#     rastr1 = cv2.warpAffine(rastr, transform_matrix, (cols, rows))
#     c = np.dstack([normalize(rastr11), normalize(rastr1), normalize(rastr1)])
#     plt.imshow(c)
#     plt.show()


# Блок дроче-Функций
def fire_landsat(way):
    channels = get_names_landsat(way)
    B7 = gdal.Open(channels["B7"]).ReadAsArray().astype('float32')
    B6 = gdal.Open(channels["B6"]).ReadAsArray().astype('float32')
    B5 = gdal.Open(channels["B5"]).ReadAsArray().astype('float32')
    B1 = gdal.Open(channels["B1"]).ReadAsArray().astype('float32')
    R75 = B7 / B5
    R76 = B7 / B6
    fire = np.zeros((B5.shape[0], B5.shape[1]))
    count = 0
    f = np.logical_and((B7 / B5 > 2.5), (B7 - B5 > 0.3))
    f2 = np.logical_and((f), (B7 > 0.5))
    f1 = np.logical_and((B7 / B5 > 1.8), (B7 - B5 > 0.17))
    f3 = np.logical_and(f2, f1)
    f10 = np.logical_and((B6 > 0.8), (B1 < 0.2))
    f11 = np.logical_and((B5 > 0.4), (B7 < 0.1))
    f13 = np.logical_and(f10, f11)
    np.logical_or(f3, f13, out=fire)
    fire_cords = np.where(fire == 1)
    print('startToCum')
    start = time()
    for x1 in range(len(fire_cords[0])):
        x = fire_cords[0][x1] - 30
        y = fire_cords[1][x1] - 30
        square75 = R75[x:x + 61, y:y + 61]
        square7 = B7[x:x + 61, y:y + 61]
        srkv = np.std(square75)
        srkvP7 = np.std(square7)
        f = np.logical_and(
            np.logical_and((R75 > (R75 + max((srkv * 3), (0.8)))), (B7 > B7 + (max((srkvP7 * 3), 0.08)))), R76 > 1.6,
            out=fire)
    print('CumToCum', (time() - start))
    # for i in range(B5.shape[0]):
    #     for j in range(B5.shape[1]):
    #         if(((B7[i][j] / B5[i][j]) > 2.5) and (B7[i][j] - B5[i][j] > 0.3) and B7[i][j] > 0.5) and (((B7[i][j] / B5[i][j]) >  1.8) and (B7[i][j] - B5[i][j] > 0.17)):
    #             count += 1
    #             fire[i][j] = 1
    #             continue
    #         if(B6[i][j] > 0.8 and B1[i][j] < 0.2 and (B5[i][j] > 0.4 or B7[i][j] < 0.1)):
    #             count += 1
    #             fire[i][j] = 1
    #         print(i,j)
    plt.imshow(fire)
    plt.show()
    return count


# Координаты модиса
def get_lat_lon(way):
    return [SD(way).select('Latitude')[:], SD(way).select('Longitude')[:]]


def get_pixel_coordinates(x, y, way):
    latLon = get_lat_lon(way)
    lat = latLon[0][x // 5][y // 5]
    lon = latLon[1][x // 5][y // 5]
    lat1 = latLon[0][(x // 5) + 1][(y // 5) + 1]
    lon1 = latLon[1][(x // 5) + 1][(y // 5) + 1]
    xlan = lat + (((lat1 - lat) / 5) * (x % 5))
    longitude = lon + (((lon1 - lon) / 5) * (x * 5))
    return lat, lon


# Ещё одна дрочефункция
def get_L(way, chanel):
    scales = get_support_data(chanel, way)
    sl = get_fileName(chanel, way)["22"]
    l = np.zeros((sl.shape[0], sl.shape[1]))
    for i in range(sl.shape[0]):
        for j in range(sl.shape[1]):
            b = float(sl[i][j]) - scales["radiance_offset"][0]
            l[i][j] = float(scales["radiance_scales"][0]) * b / 65535
    return l


def mass_cast(mas, width, long):
    m = np.zeros((width, long))
    k = width // mas.shape[0] + 1
    for y in range(width):
        for x in range(long):
            m[y][x] = mas[y // 5][x // 5]
    return m


# Это писал индус, для получения путей каналов модиса
def get_SubFileName(way, chanel):
    a = str(gdal.Info(way)).split('\n')
    name = ''
    for i in range(len(a)):
        if chanel in a[i] and 'NAME=' in a[i] and len(a[i]) == a[i].find(chanel) + len(chanel):
            name = a[i]
    return name[name.find('=') + 1:]


def get_ESUN_and_distance_sun(way):
    Info_way = gdal.Info(way).split("\n")
    Esun = ''
    Distance = ''
    for i in range(len(Info_way)):
        if Info_way[i].count("Solar Irradiance on RSB Detectors over pi=") != 0:
            Esun = Info_way[i][len("Solar Irradiance on RSB Detectors over pi=") + 2:].split(",")
        elif Info_way[i].count("Earth-Sun Distance=") != 0:
            Distance = Info_way[i][len("Earth-Sun Distance=") + 2:].split(",")
    es = [float(x) for x in Esun]
    dis = [float(x) for x in Distance]
    Esun_and_distance = {}
    Esun_and_distance["ESUN"] = es
    Esun_and_distance["distance_to_sun"] = dis
    return Esun_and_distance


def get_reflectance_scales_and_offsets(way, chanel):
    mAss = gdal.Info(get_SubFileName(way, chanel)).split('\n')
    ref_scales = ""
    ref_offset = ""
    for i in range(len(mAss)):
        if mAss[i].count("reflectance_scales=") != 0:
            ref_scales = mAss[i][len('reflectance_scales=') + 2:].split(',')
        elif mAss[i].count("reflectance_offsets=") != 0:
            ref_offset = mAss[i][len('reflectance_offsets=') + 2:].split(',')
    ref = {}
    ref["reflectance_scales"] = [float(x) for x in ref_scales]
    ref["reflectance_offsets"] = [float(x) for x in ref_offset]
    return ref


# просто вывод массива
def mas_output(mas):
    for i in range(mas.shape[0]):
        for j in range(mas.shape[1]):
            print(mas[i][j], end=' ')
        print()


# Рудимент
def normalize(input_band):
    min_a, max_a = input_band.min() * 1.0, input_band.max() * 1.0
    return ((input_band * 1.0 - min_a * 1.0) / (max_a * 1.0 - min_a))


# это вроде тоже для модиса
def get_rastr(way):
    gdalData = gdal.Open(way)
    raster = gdalData.ReadAsArray()
    mas = np.array(raster)
    print(type(mas))
    return mas


# я хз, зачем это нам
def ndvi_g(red_way, nir_way, way=0, show=True):
    if way:
        r = gdal.Open(way)
        red = r.GetRasterBand(1).ReadAsArray()
        nir = r.GetRasterBand(2).ReadAsArray()
    else:
        red = gdal.Open(red_way).ReadAsArray()
        nir = gdal.Open(nir_way).ReadAsArray()
    np.seterr(divide='ignore', invalid='ignore')
    print(red)
    ndvi_ = (nir.astype(float) - red.astype(float)) / (nir + red)
    if show:
        plt.imshow(np.dstack(ndvi_)[0])
        plt.show()
    return ndvi_


# Имба, без этой функции все сломается
def get_longitude_latitude():
    print(1)


# обрезка растра
def check_borders(longitude, latitude, level=4, region_name="Krasnodar"):
    shape = ogr.Open(
        f"/Users/kirilllesniak/Downloads/Адм_территориальные_границы_РФ_в_формате_SHP/admin_level_{level}.shp")
    indexedLayer = shape.GetLayerByIndex(0)
    region = region_name
    regIndex = 0
    for i in range(len(indexedLayer)):
        if str(indexedLayer[i].GetField('name_de')).count(region) != 0:
            regIndex = i
    feature = indexedLayer.GetFeature(regIndex)
    regGeom = feature.GetGeometryRef()
    regionMass = regGeom.ExportToWkt()[:-2][10:].replace('(', '').replace(')', '').split(',')
    a = []
    for el in regionMass:
        c = [float(el[0:el.find(" ")]), float(el[el.find(" "):])]
        a.append(c)
    point = Point(latitude, longitude)
    polygon = Polygon(a)
    # plt.show()
    return polygon.contains(point)


# ещё один ndvi
def ndvi_mas(nir, red, show=True):
    np.seterr(divide='ignore', invalid='ignore')
    print(nir[1][3] - red[1][3], nir[1][3] + red[1][3])
    ndvi = (nir - red) / (nir + red)
    if show:
        plt.imshow(ndvi)
        plt.text(0.0, 0.0, "maxNdvi:   " + str(ndvi.max()))
        plt.show()
    return ndvi


# Функция, которую кирилл искал, как сделать(вывод rgb)
def show_as_png(way):
    mas = gdal.Open(way)
    green = mas.GetRasterBand(4).ReadAsArray()
    blue = mas.GetRasterBand(3).ReadAsArray()
    red = mas.GetRasterBand(1).ReadAsArray()
    rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])
    plt.imshow(rgb)
    plt.show()


# Кирилл это сделал для модиса, но даже там это не надо
def open_file(way):
    return SD(way, SDC.WRITE | SDC.CREATE | SDC.READ)


# тоже для модиса, но полезно
def get_fileName(chanel, way):
    name = get_SubFileName(way, chanel)
    BandsArray = gdal.Open(name).ReadAsArray()
    if str(gdal.Info(name)).count('band_name') != 0:
        HelpA = str(gdal.Info(name)).split('\n')
        BandNames = ''
        for i in range(len(HelpA)):
            if HelpA[i].count("band_names") != 0:
                BandNames = HelpA[i][HelpA[i].find('=') + 1:].split(',')
        Bands = {}
        for i in range(len(BandNames)):
            Bands[BandNames[i]] = BandsArray[i]
        return Bands
    else:
        return BandsArray


def get_support_data(channel, way):
    st = gdal.Info(way)[gdal.Info(way).find("Subdatasets:") + len(channel): gdal.Info(way).find("Corner Coordinates")]
    mas = ""
    for i in range(st.find(channel) + len(channel) - 1, 0, -1):
        if st[i] != " ":
            mas += st[i]
        else:
            break
    mas1 = mas[::-1][mas[::-1].find("=") + 1:]
    mas1 = mas1.replace('"', "")
    mas = gdal.Info(mas1)
    r_scales = mas[mas.find("radiance_scales"):mas.find("\n", mas.find("radiance_scales"))]
    r_offset = mas[mas.find("radiance_offset"):mas.find("\n", mas.find("radiance_offset"))]
    radiance_scales = [float(x) for x in r_scales[r_scales.find("=") + 1:].split(",")]
    radiance_offset = [float(x) for x in r_offset[r_offset.find("=") + 1:].split(",")]
    spisok = {}
    spisok["radiance_scales"] = radiance_scales
    spisok["radiance_offset"] = radiance_offset
    return spisok


# очередная дрочефункция, но подрочевее
def fire(file_name, channel=31):
    np.seterr(divide='ignore', invalid='ignore')
    h = 6.62607015 * 10 ** -34
    c = 299792458.0
    pi = 3.141592
    c2 = 1.4387752 * 10 ** 4
    c1b = 1.19104282 * 10 ** 8
    k = 1.380649 * 10 ** -23
    lenght_wave_31 = 11.03
    lenght_wave_21 = 3.96
    lenght_wave_32 = 12.02
    lenght_wave_22 = 3.96
    rastr_21 = get_fileName("1KM_Emissive", file_name)["21"]
    rastr_31 = get_fileName("1KM_Emissive", file_name)["31"]
    rastr_22 = get_fileName("1KM_Emissive", file_name)["22"]
    rastr_32 = get_fileName("1KM_Emissive", file_name)["32"]
    rastr_250 = get_fileName("EV_250_Aggr1km_RefSB", file_name)['1']
    support_data = get_support_data("1KM_Emissive", file_name)
    channels = get_fileName("1KM_Emissive", file_name)
    keys = list(channels.keys())
    reflectans_scales_21 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_scales"][
        21 - int(keys[0])]  # костыль
    reflectans_offset_21 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_offsets"][
        21 - int(keys[0])]
    reflectans_scales_31 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_scales"][
        31 - int(keys[1])]
    reflectans_offset_31 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_offsets"][
        31 - int(keys[1])]
    reflectans_offset_250 = \
        get_reflectance_scales_and_offsets(file_name, "EV_250_Aggr1km_RefSB")["reflectance_offsets"][0]
    reflectans_scales_250 = get_reflectance_scales_and_offsets(file_name, "EV_250_Aggr1km_RefSB")["reflectance_scales"][
        0]
    radiance_scales_250 = support_data["radiance_scales"][0]
    radiance_scales_21 = support_data["radiance_scales"][21 - int(keys[0])]
    radiance_offset_21 = support_data["radiance_offset"][21 - int(keys[0])]
    radiance_scales_31 = support_data["radiance_scales"][31 - int(keys[1])]
    radiance_offset_31 = support_data["radiance_offset"][31 - int(keys[1])]
    radiance_scales_32 = support_data["radiance_scales"][32 - int(keys[1])]
    radiance_scales_22 = support_data["radiance_scales"][22 - int(keys[0])]
    radiance_offset_22 = support_data["radiance_offset"][22 - int(keys[0])]
    reflectans_scales_22 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_scales"][
        22 - int(keys[0])]
    reflectans_offset_22 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_offsets"][
        22 - int(keys[0])]
    reflectans_offset_250 = \
        get_reflectance_scales_and_offsets(file_name, "EV_250_Aggr1km_RefSB")["reflectance_offsets"][0]
    reflectans_scales_32 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_scales"][
        32 - int(keys[1])]
    reflectans_offset_32 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_offsets"][
        32 - int(keys[1])]
    SolarZenith = mass_cast(get_fileName("SolarZenith", file_name), rastr_22.shape[0], rastr_22.shape[1])
    ESUN = get_ESUN_and_distance_sun(file_name)["ESUN"][0]
    distance_to_sun = get_ESUN_and_distance_sun(file_name)["distance_to_sun"][0]
    ir_21 = np.zeros((rastr_21.shape[0], rastr_21.shape[1]))
    ir_31 = np.zeros((rastr_31.shape[0], rastr_31.shape[1]))
    ir_250 = np.zeros((rastr_250.shape[0], rastr_250.shape[1]))
    ir_22 = np.zeros((rastr_22.shape[0], rastr_22.shape[1]))
    ir_32 = np.zeros((rastr_32.shape[0], rastr_32.shape[1]))
    canall_21 = np.zeros((rastr_21.shape[0], rastr_21.shape[1]))
    canall_31 = np.zeros((rastr_31.shape[0], rastr_31.shape[1]))
    canall_22 = np.zeros((rastr_22.shape[0], rastr_22.shape[1]))
    canall_32 = np.zeros((rastr_32.shape[0], rastr_32.shape[1]))
    data = np.zeros((rastr_22.shape[0], rastr_22.shape[1]))  # 22 канал используем как базовы
    rastr_fire = np.zeros((rastr_22.shape[0], rastr_22.shape[1]))
    count_bitie = 0
    for i in range(ir_22.shape[0]):
        for j in range(ir_22.shape[1]):
            if rastr_22[i][j] != 65534:
                ir_21[i][j] = (radiance_scales_21 * (rastr_21[i][j] - reflectans_offset_21))
                ir_31[i][j] = (radiance_scales_31 * (rastr_31[i][j] - reflectans_offset_31))
                ir_250[i][j] = (radiance_scales_250 * (rastr_250[i][j] - reflectans_scales_250))
                ir_22[i][j] = (radiance_scales_22 * (rastr_22[i][j] - reflectans_offset_22))
                ir_32[i][j] = (radiance_scales_32 * (rastr_32[i][j] - reflectans_offset_32))
            else:
                count_bitie += 1
                rastr_22[i][j] = -1
    for i in range(rastr_22.shape[0]):
        for j in range(rastr_22.shape[1]):
            if rastr_22[i][j] != -1:
                canall_21[i][j] = (c2 / lenght_wave_21) / (math.log(1 + ((c1b * lenght_wave_21 ** -5) / ir_21[i][j])))
                canall_31[i][j] = (c2 / lenght_wave_31) / (math.log1p(1 + ((c1b * lenght_wave_31 ** -5) / ir_31[i][j])))
                data[i][j] = abs(canall_21[i][j] - canall_31[i][j])
                if ir_22[i][j] >= 0.74 and 6.9 <= ir_32[i][j] <= 10 and data[i][j] >= 12 and ir_250[i][j] < 0.112:
                    rastr_fire[i][j] = 1
                    lat, lon = get_pixel_coordinates(i, j, file_name)
                    print(lat, lon)
                    print(check_borders(lat, lon, region_name="Kaliningrad"))
            else:
                pass
    return rastr_fire
    # print(rastr[i][j])
    # valid_range - битые пиксели
    # ширина и долгота записаны в отдельном файле и имееют привизку к каждому пятому пикселю
    # rastr[i][j] = lenght_wave/math.cos(math.pi*SolarZenith[i][j]/180)
    # 859nm / cos(3.14*solarZenith/180) выравнивание значений
    # print(rastr[i][j], end=" ")
    # print()


import numpy as np


def f(x_1, y_1, x_2, y_2, coord1, coord2):
    return np.argmin(np.abs(x_2 - np.full(x_2.shape, x_1[coord1]))), np.argmin(
        np.abs(y_2 - np.full(y_2.shape, y_1[coord2])))


def main():
    ways = {"mod14": "/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD14.hdf",
            "mod3": "/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD03.hdf",
            "mod2": "/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD021KM.hdf",
            "mod2_1km": "HDF4_EOS:EOS_SWATH:/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD021KM.hdf:MODIS_SWATH_Type_L1B:EV_1KM_RefSB_Uncert_Indexes",
            # "mod021_1km" :"HDF4_EOS:EOS_SWATH:/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD021KM.hdf:MODIS_SWATH_Type_L1B:EV_1KM_Emissive",
            "mod03": "/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD03.hdf",
            "mod021_astrahan": "/Users/kirilllesniak/Downloads/hdf-sort/1/20220310_092621_TERRA_MOD021KM.hdf",
            "mod021_kaliningrad": "/Users/kirilllesniak/Downloads/hdf-sort/1/20220310_092621_TERRA_MOD021KM.hdf",
            "landsat_astr": "/Users/kirilllesniak/Downloads/LC09_L2SP_168028_20220321_20220323_02_T1",
            "landsat_4": "/Users/kirilllesniak/Downloads/Landsat 8 2017/LC08_L2SP_119016_20170815_20200903_02_T1_SR_B4.TIF",
            "landsat_5": "/Users/kirilllesniak/Downloads/Landsat 8 2017/LC08_L2SP_119016_20170815_20200903_02_T1_SR_B5.TIF",
            "landsat_red": "/Users/kirilllesniak/Downloads/Landsat 8 2017/LC08_L2SP_119016_20170815_20200903_02_T1_SR_B4.TIF",
            "landsat_green": "/Users/kirilllesniak/Downloads/Landsat 8 2017/LC08_L2SP_119016_20170815_20200903_02_T1_SR_B3.TIF",
            "landsat_blue": "/Users/kirilllesniak/Downloads/Landsat 8 2017/LC08_L2SP_119016_20170815_20200903_02_T1_SR_B2.TIF", }
    yuras_ways = {
        'land_astrahan': "C:/Users/perminov_u/Downloads/Telegram Desktop/LC09_L2SP_168028_20220321_20220323_02_T1",
        'sentinelZip': "C:/Users/perminov_u/Downloads/Telegram Desktop/S2B_MSIL1C_20230211T044929_N0509_R076_T44QRJ_20230211T064447_SAFE.zip",
        'sentinel': "C:/Users/perminov_u/Downloads/S2B_MSIL1C_20230211T044929_N0509_R076_T44QRJ_20230211T064447.SAFE",
        'anotherSentinel': "C:/Users/perminov_u/Downloads/S2A_MSIL1C_20230218T085021_N0509_R107_T37VDG_20230218T093702.SAFE",
        'sentinel_spb': "C:/Users/perminov_u/Downloads/Новая папка/S2B_MSIL1C_20230301T090819_N0509_R050_T35UQS_20230301T094840.SAFE",
        'sentinel_spb_1': "C:/Users/perminov_u/Downloads/Новая папка/S2B_MSIL1C_20230225T093039_N0509_R136_T35VPG_20230225T100846.SAFE",
        'sentinel_Kyiv': "C:/Users/perminov_u/Downloads/Новая папка/S2B_MSIL1C_20230301T090819_N0509_R050_T36UUB_20230301T094840.SAFE",
        'sentinel_Kyiv_1': "C:/Users/perminov_u/Downloads/Новая папка/S2B_MSIL1C_20230225T093039_N0509_R136_T36VUM_20230225T100846.SAFE",
        'sentinel_oral': "C:/Users/perminov_u/Downloads/S2A_MSIL2A_20220401T065621_N0400_R063_T40TGS_20220401T095213.SAFE",
        'sentinel_oral1': "C:/Users/perminov_u/Downloads/S2A_MSIL2A_20221018T065901_N0400_R063_T41TLM_20221018T102202.SAFE",
        'hach_ozero':"C:/Users/perminov_u/Downloads/S2B_MSIL2A_20221025T055919_N0400_R091_T43TDM_20221025T072143.SAFE",
        'hach_ozero1':"C:/Users/perminov_u/Downloads/S2B_MSIL1C_20230314T055639_N0509_R091_T43TDM_20230314T074739.SAFE",
        'gay_ozero':"C:/Users/perminov_u/Downloads/S2B_MSIL1C_20230218T075959_N0509_R035_T36NWF_20230220T131038.SAFE",
        'gay_ozero2':"C:/Users/perminov_u/Downloads/S2B_MSIL1C_20221031T080009_N0400_R035_T36NWF_20221031T094754.SAFE"}
    default_save_way="C:/Users/perminov_u/Desktop/"
    # print(ndvi(ways["mod3"], ways["mod2"],show=True))
    # way = "/MODIS_SWATH_Type_L1B/Geolocation Fields"
    # print(gdal.Info(gdal.Info(ways['mod2']+way)))
    # fire(ways["mod021_kaliningrad"])
    # fire_landsat(yuras_ways['land_astrahan'])
    # print(sentinel_ndsi(yuras_ways['sentinel']))
    # print(sentinel_mndwi(yuras_ways['sentinel']))
    # whater_difference(yuras_ways['sentinel'], yuras_ways['anotherSentinel'])
    # sentinel_mndwi(yuras_ways['sentinel_oral'])
    # a=f(get_coords_list(open_clean_band(get_ways_sentinel(yuras_ways['sentinel_oral'])['B03']))['x'],get_coords_list(open_clean_band(get_ways_sentinel(yuras_ways['sentinel_oral'])['B03']))['y'],get_coords_list(open_clean_band(get_ways_sentinel(yuras_ways['sentinel_oral1'])['B03']))['x'],get_coords_list(open_clean_band(get_ways_sentinel(yuras_ways['sentinel_oral1'])['B03']))['y'],sentinel_mndwi(yuras_ways['sentinel_oral']), sentinel_mndwi(yuras_ways['sentinel_oral1']))
    # print(a)
    # sentinel_ndgr(yuras_ways['sentinel_spb'])
    # a=get_coords_list(open_clean_band(yuras_ways['land_astrahan']))['x']
    # a1=get_coords_list(open_clean_band(get_ways_sentinel(yuras_ways['sentinel_oral'])['B08']))['x']
    # print(a)
    # print(a1)
    # start = time()
    # sentinel_cloud(yuras_ways['sentinel_oral1'])
    # sentinel_cloud(yuras_ways["sentinel_Kyiv"])
    save_as_tiff(whater_difference(yuras_ways['sentinel_oral1'], yuras_ways['sentinel_oral']),default_save_way, "sentinel_oral_ndvi")
    # cords = sentinel_coordinates(yuras_ways['sentinel_oral1'])
    # print(cords, cords1)
    # shot1 = sentinel_coordinates(yuras_ways['sentinel_oral1'])
    # shot2 = sentinel_coordinates(yuras_ways['sentinel_oral'])
    # print(f(shot1['lat'][0],shot1['lon'][0], shot2['lat'][0], shot2['lon'][0], 46.88962509, 59.56566092))
    # print(shot2['lat'])
    # print(shot2['lon'])
    # print(shot1['lat'])
    # print(shot1['lon'])

    # sentinel_cloud(yuras_ways['sentinel_oral1'])
    # sentinel_mndwi(yuras_ways['sentinel_oral'])
    # sentinel_mndwi(yuras_ways['sentinel_oral1'])
    # print(np.max(np.array(get_names_sentinel(yuras_ways['sentinel'], 'GREEN'))))
    # print(fire_landsat(yuras_ways['land_astrahan']))
    # get_L(ways['mod2'], 'EV_1KM_Emissive')
    # gdalData = gdal.Open(ways["mod2"])


if __name__ == '__main__':
    main()
    # воимя отца сына и святого дуба помоги этому коду заработаать
