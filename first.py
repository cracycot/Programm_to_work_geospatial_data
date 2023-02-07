from pyhdf import HDF
from pyhdf.SD import SD,SDC
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
import h5py




import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.plotting import plot_polygon, plot_points


def get_L(way, chanel):
    scales=get_support_data(chanel,way)
    sl=get_fileName(chanel, way)["22"]
    l = np.zeros((sl.shape[0] ,sl.shape[1] ))
    for i in range(sl.shape[0]):
        for j in range(sl.shape[1]):
            b = float(sl[i][j])- scales["radiance_offset"][0]
            l[i][j]=float(scales["radiance_scales"][0])*b/65535
    return l

def mass_cast(mas,width , long):
    m = np.zeros((width, long))
    k = width // mas.shape[0] + 1
    for y in range(width):
        for x in range(long):
            m[y][x] = mas[y// 5][x//5]
    return m

def get_SubFileName(way, chanel):
    a = str(gdal.Info(way)).split('\n')
    name = ''
    for i in range(len(a)):
        if chanel in a[i] and 'NAME=' in a[i] and len(a[i]) == a[i].find(chanel) + len(chanel):
            name = a[i]
    return name[name.find('=') + 1:]

def get_ESUN_and_distance_sun(way):
    Info_way = gdal.Info(way).split("\n")
    Esun=''
    Distance = ''
    for i in range(len(Info_way)):
        if Info_way[i].count("Solar Irradiance on RSB Detectors over pi=")!=0:
            Esun=Info_way[i][len("Solar Irradiance on RSB Detectors over pi=")+2:].split(",")
        elif Info_way[i].count("Earth-Sun Distance=")!=0:
            Distance = Info_way[i][len("Earth-Sun Distance=") + 2 :].split(",")
    es = [float(x) for x in Esun]
    dis = [float(x) for x in Distance]
    Esun_and_distance = {}
    Esun_and_distance["ESUN"] = es
    Esun_and_distance["distance_to_sun"] = dis
    return Esun_and_distance

def get_reflectance_scales_and_offsets(way,chanel):
    mAss=gdal.Info(get_SubFileName(way,chanel)).split('\n')
    ref_scales = ""
    ref_offset = ""
    for i in range(len(mAss)):
        if mAss[i].count("reflectance_scales=")!=0:
            ref_scales = mAss[i][len('reflectance_scales=')+2:].split(',')
        elif mAss[i].count("reflectance_offsets=")!=0:
            ref_offset = mAss[i][len('reflectance_offsets=')+2:].split(',')
    ref = {}
    ref["reflectance_scales"] = [float(x) for x in ref_scales]
    ref["reflectance_offsets"] = [float(x) for x in ref_offset]
    return ref
def mas_output(mas):
    for i in range(mas.shape[0]):
        for j in range(mas.shape[1]):
            print(mas[i][j], end=' ')
        print()
def normalize(input_band):
    min_a , max_a = input_band.min()*1.0 ,input_band.max()*1.0
    return ((input_band*1.0 - min_a*1.0)/(max_a*1.0 - min_a))
def get_rastr(way):
    gdalData = gdal.Open(way)
    raster = gdalData.ReadAsArray()
    mas = np.array(raster)
    print(type(mas))
    return mas
def ndvi(red_way, nir_way,way = 0, show=True):
    if way:
        r = gdal.Open(way)
        red = r.GetRasterBand(1).ReadAsArray()
        nir = r.GetRasterBand(2).ReadAsArray()
    else:
        red = gdal.Open(red_way).ReadAsArray()
        nir =  gdal.Open(nir_way).ReadAsArray()
    np.seterr(divide='ignore', invalid='ignore')
    print(red)
    ndvi_ = (nir.astype(float) - red.astype(float))/(nir + red)
    if show:
        plt.imshow(np.dstack(ndvi_)[0])
        plt.show()
    return ndvi_
def get_longitude_latitude():
    print(1)
def check_borders(longitude, latitude, level=4, region_name = "Krasnodar"):
    shape = ogr.Open(f"/Users/kirilllesniak/Downloads/Адм_территориальные_границы_РФ_в_формате_SHP/admin_level_{level}.shp")
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
    plot_polygon(polygon)
    plt.show()
    return polygon.contains(point)

def ndvi_mas(nir, red, show=True):
    np.seterr(divide='ignore', invalid='ignore')
    print(nir[1][3] - red[1][3], nir[1][3] + red[1][3])
    ndvi=(nir - red) / (nir + red)
    if show:
        plt.imshow(ndvi)
        plt.text(0.0, 0.0, "maxNdvi:   "+str(ndvi.max()))
        plt.show()
    return ndvi
def show_as_png(way):
    mas = gdal.Open(way)
    green = mas.GetRasterBand(4).ReadAsArray()
    blue = mas.GetRasterBand(3).ReadAsArray()
    red = mas.GetRasterBand(1).ReadAsArray()
    rgb = np.dstack([normalize(red),normalize(green),normalize(blue)])
    plt.imshow(rgb)
    plt.show()

def open_file(way):
    return SD(way,SDC.WRITE|SDC.CREATE|SDC.READ)

def get_fileName(chanel, way):
    name=get_SubFileName(way, chanel)
    BandsArray= gdal.Open(name).ReadAsArray()
    if str(gdal.Info(name)).count('band_name')!=0:
        HelpA=str(gdal.Info(name)).split('\n')
        BandNames=''
        for i in range(len(HelpA)):
            if HelpA[i].count("band_names")!=0:
                BandNames=HelpA[i][HelpA[i].find('=')+1:].split(',')
        Bands={}
        for i in range(len(BandNames)):
            Bands[BandNames[i]]=BandsArray[i]
        return Bands
    else:
        return BandsArray
def get_support_data(channel, way):
    st = gdal.Info(way)[gdal.Info(way).find("Subdatasets:")+ len(channel) : gdal.Info(way).find("Corner Coordinates")]
    mas = ""
    for i in range(st.find(channel) + len(channel)- 1 , 0 , -1):
        if st[i] != " ":
            mas += st[i]
        else:
            break
    mas1 = mas[::-1][mas[::-1].find("=")+1:]
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
def fire(file_name, channel = 31):
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
    support_data = get_support_data("1KM_Emissive",file_name)
    channels=get_fileName("1KM_Emissive",file_name)
    keys=list(channels.keys())
    reflectans_scales_21 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_scales"][21 - int(keys[0])] # костыль
    reflectans_offset_21 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_offsets"][21 - int(keys[0])]
    reflectans_scales_31 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_scales"][31 - int(keys[1])]
    reflectans_offset_31 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_offsets"][31 - int(keys[1])]
    reflectans_offset_250 = get_reflectance_scales_and_offsets(file_name, "EV_250_Aggr1km_RefSB")["reflectance_offsets"][0]
    reflectans_scales_250 = get_reflectance_scales_and_offsets(file_name, "EV_250_Aggr1km_RefSB")["reflectance_scales"][0]
    radiance_scales_250 = support_data["radiance_scales"][0]
    radiance_scales_21 = support_data["radiance_scales"][21 - int(keys[0])]
    radiance_offset_21 = support_data["radiance_offset"][21 - int(keys[0])]
    radiance_scales_31 = support_data["radiance_scales"][31 - int(keys[1])]
    radiance_offset_31 = support_data["radiance_offset"][31 - int(keys[1])]
    radiance_scales_32 = support_data["radiance_scales"][32 - int(keys[1])]
    radiance_scales_22 = support_data["radiance_scales"][22 - int(keys[0])]
    radiance_offset_22 = support_data["radiance_offset"][22 - int(keys[0])]
    reflectans_scales_22 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_scales"][22 - int(keys[0])]
    reflectans_offset_22 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_offsets"][22 - int(keys[0])]
    reflectans_offset_250 = get_reflectance_scales_and_offsets(file_name, "EV_250_Aggr1km_RefSB")["reflectance_offsets"][0]
    reflectans_scales_32 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_scales"][32 - int(keys[1])]
    reflectans_offset_32 = get_reflectance_scales_and_offsets(file_name, "1KM_RefSB")["reflectance_offsets"][32 - int(keys[1])]
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
    data = np.zeros((rastr_22.shape[0], rastr_22.shape[1])) # 22 канал используем как базовы
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
                #t11 = (1.4387752 * 10 ** 4 / 11.03) / ( math.log(1 + (1.19104282 * 10 ** 8 * 11.03 ** (-5)) / bands_dict['EV_1KM_Emissive_31'][i]))
                canall_31[i][j] = (c2 / lenght_wave_31) / (math.log1p(1 + ((c1b * lenght_wave_31 ** -5) / ir_31[i][j])))
                data[i][j] = abs(canall_21[i][j] - canall_31[i][j])
                if ir_22[i][j] >= 0.74 and 6.9 <= ir_32[i][j] <= 10 and data[i][j] >= 12 and ir_250[i][j] < 0.112:
                    rastr_fire[i][j] = 1
                    print(i, j)
            else:
                pass
    #mas_output(canall_31)
    #data = abs(canall_21 - canall_31)
    #print(ir_250.min())
    return rastr_fire
    # print(rastr[i][j])
    # valid_range - битые пиксели
    # ширина и долгота записаны в отдельном файле и имееют привизку к каждому пятому пикселю
    # rastr[i][j] = lenght_wave/math.cos(math.pi*SolarZenith[i][j]/180)
    # 859nm / cos(3.14*solarZenith/180) выравнивание значений
    # print(rastr[i][j], end=" ")
    # print()
def main():
    ways = {"mod14" : "/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD14.hdf",
    "mod3" : "/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD03.hdf" ,
    "mod2" : "/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD021KM.hdf",
    "mod2_1km" : "HDF4_EOS:EOS_SWATH:/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD021KM.hdf:MODIS_SWATH_Type_L1B:EV_1KM_RefSB_Uncert_Indexes",
    #"mod021_1km" :"HDF4_EOS:EOS_SWATH:/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD021KM.hdf:MODIS_SWATH_Type_L1B:EV_1KM_Emissive",
    "mod03":"/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD03.hdf",
    "mod021_astrahan":"/Users/kirilllesniak/Downloads/hdf-sort/1/20220310_092621_TERRA_MOD021KM.hdf",
    "mod021_kaliningrad":"/Users/kirilllesniak/Downloads/hdf-sort/1/20220310_092621_TERRA_MOD021KM.hdf",
    "landsat_4" : "/Users/kirilllesniak/Downloads/Landsat 8 2017/LC08_L2SP_119016_20170815_20200903_02_T1_SR_B4.TIF",
    "landsat_5" : "/Users/kirilllesniak/Downloads/Landsat 8 2017/LC08_L2SP_119016_20170815_20200903_02_T1_SR_B5.TIF",
    "landsat_red" : "/Users/kirilllesniak/Downloads/Landsat 8 2017/LC08_L2SP_119016_20170815_20200903_02_T1_SR_B4.TIF",
    "landsat_green" : "/Users/kirilllesniak/Downloads/Landsat 8 2017/LC08_L2SP_119016_20170815_20200903_02_T1_SR_B3.TIF",
    "landsat_blue" : "/Users/kirilllesniak/Downloads/Landsat 8 2017/LC08_L2SP_119016_20170815_20200903_02_T1_SR_B2.TIF",}
    #print(ndvi(ways["mod3"], ways["mod2"],show=True))
    way = "/MODIS_SWATH_Type_L1B/Geolocation Fields"
    print(gdal.Info(gdal.Info(ways['mod2']+way)))
    #fire(ways["mod021_kaliningrad"])
    #get_L(ways['mod2'], 'EV_1KM_Emissive')
    #gdalData = gdal.Open(ways["mod2"])
if __name__ == '__main__':
    main()
    #воимя отца сына и святого дуба помогии этому коду заработаать