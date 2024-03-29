
from pyhdf.SD import SD, SDC
from osgeo import gdal
from osgeo import ogr
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
from time import time
import xarray
import cv2
from PIL import Image

def save_as_tiff(massive, way, name):
    Image.fromarray(massive).save(way+name+".tif")
# получение координат для sentinel, 
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


def get_cordinates_sentinel(file, x, y):
    return [float(file[x][y].coords['x']), float(file[x][y].coords['y'])]


# получение широты и долготы для 
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


# разница водного индекса, но при желании можно перековырять под другой
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


# расчет изменения водной поверхности и получения результата в формате TIFF файла
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
    print(cords)
    print(corns1)
    print(points)
    rastr = gdal.Open(get_ways_sentinel(way)['B08']).ReadAsArray()
    rastr11 = gdal.Open(get_ways_sentinel(way1)['B08']).ReadAsArray()
    rows, cols = rastr.shape[:2]
    transform_matrix = cv2.getAffineTransform(np.float32([[0, 0], [rows - 1, 0], [0, cols - 1]]), np.float32(points))
    rastr1 = cv2.warpAffine(rastr, transform_matrix, (cols, rows))
    c = np.dstack([normalize(rastr11), normalize(rastr1), normalize(rastr1)])
    plt.imshow(c)
    plt.show()


# Поиск потенциальных пикселей, в которых предполагается возгарание
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
    print('Начало работы алгоритма')
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
    print('Время работы', (time() - start))
    plt.imshow(fire)
    plt.show()
    return count


# получение координат из hdf файла с сенсора MODIS
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


# Получения вспомогательного значения
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


# Функция для получения путей до конкретных спектральных каналов модиса
def get_SubFileName(way, chanel):
    a = str(gdal.Info(way)).split('\n')
    name = ''
    for i in range(len(a)):
        if chanel in a[i] and 'NAME=' in a[i] and len(a[i]) == a[i].find(chanel) + len(chanel):
            name = a[i]
    return name[name.find('=') + 1:]

# Получение значений угла падения солнечных лучей и растояния спуитника от солнца (для более корректного расчета индексов)
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

# получение вспомогательных значений для использовании в формуле Планка
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


#Вывод массива
def mas_output(mas):
    for i in range(mas.shape[0]):
        for j in range(mas.shape[1]):
            print(mas[i][j], end=' ')
        print()

#нормализация до значений пикселей от 0.0 до 1.0
def normalize(input_band):
    min_a, max_a = input_band.min() * 1.0, input_band.max() * 1.0
    return ((input_band * 1.0 - min_a * 1.0) / (max_a * 1.0 - min_a))


# Поучения растра системы MODIS
def get_rastr(way):
    gdalData = gdal.Open(way)
    raster = gdalData.ReadAsArray()
    mas = np.array(raster)
    print(type(mas))
    return mas
    
#Расчет индекса ndvi
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




# Функция вывода rgb
def show_as_png(way):
    mas = gdal.Open(way)
    green = mas.GetRasterBand(4).ReadAsArray()
    blue = mas.GetRasterBand(3).ReadAsArray()
    red = mas.GetRasterBand(1).ReadAsArray()
    rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])
    plt.imshow(rgb)
    plt.show()


#Вспомогательная функция для нахождения пути до файла
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


# Вывод пожарных пикселей
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




def f(x_1, y_1, x_2, y_2, coord1, coord2):
    return np.argmin(np.abs(x_2 - np.full(x_2.shape, x_1[coord1]))), np.argmin(
        np.abs(y_2 - np.full(y_2.shape, y_1[coord2])))


#рассчет индекса mndwi для рассчета изменения водной поверхности
def sentinel_mndwi_for_same_shots_with_connection(way1, way2):
    ass1=gdal.Open(way1).ReadAsArray()
    ass2=gdal.Open(way2).ReadAsArray()
    mndwi_sentinel1 = (ass1[2] - ass1[11]) / (ass1[2] + ass1[11])
    mndwi_sentinel2 = (ass2[2] - ass2[11]) / (ass2[2] + ass2[11])
    mndwi_sentinel=np.hstack([mndwi_sentinel1, mndwi_sentinel2])
    mndwi_sentinel[mndwi_sentinel<0]=0
    mndwi_sentinel[mndwi_sentinel>0]=1
    mndwi_sentinel[np.logical_and((mndwi_sentinel!=0), (mndwi_sentinel!=1))]=0
    # plt.imshow(mndwi_sentinel)
    # plt.show()
    return mndwi_sentinel
def main():

    #пример рассчета изменения водной поверхности на двух снимках разной даты

    #замените ваши пути
    default_save_way="your_way_to_save"
    
    ways = {"way1" : "your_way_first", 
           "way2" : "your_way_second"}
    
    dun21=sentinel_mndwi_for_same_shots_with_connection(ways['way1'], ways['way2'])
    
    change=np.zeros((dun21.shape[0], dun21.shape[1]))
    
    change[np.logical_and((dun21==1), (dun22==1))]=2
    change[np.logical_and((dun21==1),(dun22==0))]=1
    change[np.logical_and((dun21==0), (dun22==0))]=3
    
    plt.imshow(change)
    plt.show()
    
    print(np.count_nonzero(change==1))
    print(np.count_nonzero(change==2))
    print(np.count_nonzero(change==3))
    
    save_as_tiff(change, default_save_way, 'Answer_filename')


if __name__ == '__main__':
    main()
