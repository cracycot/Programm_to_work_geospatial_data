import math
import sys
import numpy as np
from osgeo import gdal
from time import time
from osgeo import ogr
import matplotlib.pyplot as plt
from osgeo import ogr
from main import regionMass
from shapely import Polygon
from shapely.geometry.polygon import Polygon
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.plotting import plot_polygon, plot_points
# def get_ways_sentinel(way):
#     way1 = way + "/HTML" + "/GRANULE"
#     a = os.listdir(way1)
#     print(a)
#     for el in a:
#         if el != "QI_DATA":
#             way1 += el.split(".")[0]
#             way1 += "/"
#     way1 += "IMG_DATA"
#     ways = dict()
#     for el in os.listdir(way1):
#         a = 1
#         ways["B" + el[-6:-4]] = (way1 + el)
#     return ways
#6-5-4
#/Users/kirilllesniak/Downloads/S2B_MSIL1C_20230211T044929_N0509_R076_T44QRJ_20230211T064447.SAFE/HTML/GRANULE
#/Users/kirilllesniak/Downloads/S2B_MSIL1C_20230211T044929_N0509_R076_T44QRJ_20230211T064447.SAFE
#print(os.listdir("/Users/kirilllesniak/Downloads/S2B_MSIL1C_20230211T044929_N0509_R076_T44QRJ_20230211T064447.SAFE/HTML/GRANULE.DS_Store"))
def get_ways_sentinell(way):
    ways_slov = dict()
    for root,dirs,files in os.walk(way):
        for filenames in files:
            if filenames[0] == "T":
                ways_slov[filenames[-7:-4]] = root + "/" + filenames
    return ways_slov
#print(gdal.Open('/Users/kirilllesniak/Downloads/S2B_MSIL1C_20230211T044929_N0509_R076_T44QRJ_20230211T064447.SAFE/T44QRJ_20230211T044929_B01.jp2'))
print(gdal.Open(get_ways_sentinell("/Users/kirilllesniak/Downloads/S2B_MSIL1C_20230211T044929_N0509_R076_T44QRJ_20230211T064447.SAFE")["B08"]))

# start=time()
# a = gdal.Open("/Users/kirilllesniak/Downloads/S2B_MSIL1C_20230211T044929_N0509_R076_T44QRJ_20230211T064447.SAFE/HTML/GRANULE/L1C_T44QRJ_A030990_20230211T050134/IMG_DATA/T44QRJ_20230211T044929_B03.jp2").ReadAsArray()
# print(a)
# print(np.min(a), np.max(a))
# print(time()-start)

# a = []
# for el in regionMass:
#     c = [float(el[0:el.find(" ")]), float(el[el.find(" "):])]
#     a.append(c)
# point1 = Point(38,46)
# point2 = Point(40.137,44.777)
# point3 = Point(39.802,43.459)
# point4 = Point(37.582,46.419)
# polygon = Polygon(a)
# plot_polygon(polygon)
# plt.show()
# print(polygon.contains(point1))
# print(polygon.contains(point2))
# print(polygon.contains(point3))
# print(polygon.contains(point4))
# x = [1, 5, 10, 15, 20]
# y1 = [1, 7, 3, 5, 11]
# y2 = [4, 3, 1, 8, 12]
# x = np.arange(0, 10, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
# plt.savefig('saved_figure-colored.png', facecolor = 'red')
# rastr=[[0]*100 for _ in range(100)]
# def line(Ax, Ay, Bx, By):
#     xdif=Bx-Ax
#     ydif=By-Ay
#     x=Ax
#     y=Ay
#     hiplen=math.sqrt((xdif**2)+(ydif**2))
#     rastr[int(x)][int(y)] = 1
#     for i in range(int(hiplen)):
#         rastr[int(x)][int(y)]=1
#         x+=xdif/hiplen
#         y+=ydif/hiplen
# line(20, 20, 80, 20)
# line(80, 20, 80, 80)
# line(80, 80, 20, 80)
# line(20, 80, 20, 20)
# def ins(a, b):
#     if rastr[a][b]==0:
#         ins(a+1,b)
#         ins(a-1, b)
#         ins(a, b+1)
#         ins(a, b-1)
# import sys
# @l_cashe
# ins(50, 50)
# plt.imshow(rastr)
# plt.show()
