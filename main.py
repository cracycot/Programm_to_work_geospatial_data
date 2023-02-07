
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from osgeo import ogr
import matplotlib.pyplot as mpl
shape =ogr.Open("/Users/kirilllesniak/Downloads/Адм_территориальные_границы_РФ_в_формате_SHP/admin_level_4.shp")
indexedLayer = shape.GetLayerByIndex(0)
region="Krasnodar"
regIndex=0
for i in range(len(indexedLayer)):
     if str(indexedLayer[i].GetField('name_de')).count(region)!=0:
         regIndex=i
feature = indexedLayer.GetFeature(regIndex)
regGeom=feature.GetGeometryRef()
regionMass=regGeom.ExportToWkt()[:-2][10:].replace('(','').replace(')','').split(',')

rastr=[[0]*700 for _ in range(700)]
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
# def linedef(ax, ay, bx, by):
#     xdif=bx-ax
#     ydif=by-ay
#     x=ax
#     y=ay
#     hiplen=math.sqrt((xdif**2)+(ydif**2))
#     a=0
#     count=0
#     for i in range(int(hiplen)):
#
#         if rastr[int(x)][int(y)]==1:
#             if a==0:
#                 count+=1
#                 a=1
#         x+=xdif/hiplen
#         y+=ydif/hiplen
#     return count
# for i in range(len(regionMass)-1):
#     cord=regionMass[i].split(' ')
#     cord1= regionMass[i+1].split(' ')
#     x=int(cord[0][:5].replace('.', ''))-3600
#     y=int(cord[1][:5].replace('.', ''))-5000
#     x1=int(cord1[0][:5].replace('.', ''))-3600
#     y1=int(cord1[1][:5].replace('.', ''))-5000
#     line(x,y,x1,y1)
#
# cord = regGeom.Centroid().ExportToWkt()[7:][:-1].split(' ')
# x = int(cord[0][:5].replace('.', '')) - 3600
# y = int(cord[1][:5].replace('.', '')) - 5000
# for i in range(len(rastr[0])):
#     for j in range(len(rastr)):
#
#         cros=linedef(i, j, x, y)
#         if cros%2==0:
#             rastr[i][j]=2
#         print(i, j)



# plt.figure(figsize=(12, 7))
# plt.plot([0,0.25,1],[0,1,0.5])
# def ins(x1,y1):
#     x=abs(x1)
#     y=abs(y1)
#     if x>=0 and y>=0 and x<700 and y<700:
#         if rastr[x][y]==0:
#             rastr[x][y]=1
#             ins(x+1,y)
#             ins(x,y+1)
#             ins(x-1,y)
#             ins(x,y-1)



# for y in range(len(rastr)):
#     a = str(rastr[y])
#     print(a)
#     index_1 = a.find("1")
#     index_2 = a.rfind("1")
#     if index_2 != -1 and index_1 != -1:
#         for i in range(index_1, index_2):
#             rastr[y][i] = 1

# for i in range(len(rastr[0])):
#     count = 0
#     a=0
#     for j in range(len(rastr)):
#         if rastr[i][j]>0:
#             if a==0:
#                 count+=1
#             a = 1
#         else:
#             a=0
#         if count%2==1:
#             rastr[i][j]=1


# def ray(x, y, alpha):
#     count=0
#     a=0
#     Ass=0
#     Ass1=0
#     if alpha==90:
#         Ass=1
#     else:
#         Ass1=1
#     while x>0 and x<len(rastr[0]) and y>0 and y<len(rastr):
#         if rastr[x][y]==1:
#             if a==0:
#                 count+=1
#             a=1
#         if rastr[x][y]==0:
#             a=0
#         x+=Ass
#         y+=Ass1
#     return count

# for i in range(len(rastr[0])):
#     for j in range(len(rastr)):
#         if rastr[i][j]!=1:
#
#             if cross%2==1:
#                 rastr[i][j]=2
#             print(i, j)


# for i in range(len(rastr[0])):
#     for j in range(len(rastr)):
#         print(i, j)
#         a=[]
#         for x in range(4):
#             a.append(ray(i, j, x*90)%2)
#         if a.count(1)>a.count(0):
#             rastr[i][j]=2
#         elif a.count(1)==a.count(0):
#             if rastr[i+1][j]>0 and i<len(rastr[0])-1 :
#                 rastr[i][j]=2
# print('sdfsdfasdf')
# cord=regGeom.Centroid().ExportToWkt()[7:][:-1].split(' ')
# x = int(cord[0][:5].replace('.', ''))-3600
# y = int(cord[1][:5].replace('.', ''))-5000
# print(cord)
# print(x, y)
# plt.imshow(rastr)
# plt.show()
# print(regGeom.Centroid())



#print(a.geometry())
#print(help(a))
# for feature in indexedLayer:
#     print(feature.GetField('name_pt'))
'''for i in range(len(penetrate)):
    print(penetrate.GetFeature(i), i)'''
# for i in range(len(indexedLayer)):
#     feature=indexedLayer.GetFeature(i)
#     print(i, feature)
#penetrate=shape.GetLayer(0).GetFeature(0).ExportToJson()
# print(feature.GetPGLayer('Zemutinio Naugardo sritis'))
#print(gdal.Info(feature))
# print( feature.geometry())
#print(penetrate.geometry())
# layer_definition=indexedLayer.GetLayerDefn()
# for i in range(layer_definition.GetFieldCount()):
#     field_definition=layer_definition.GetFieldDefn(i)
#     print('%s (%s)'%(field_definition.name, field_definition.type), i)
# layers=indexedLayer.GetLayerDefn().GetFieldDefn(70)
#print(layers.GetPGLayer())
#     #7GetPrecision justify thisown
#     #GetFID-poluchit' id
# feat=indexedLayer.GetFeature(1)
# print(feat)
# geom = feat.GetGeometryRef()
# print ("%.3f, %.3f" % ( geom.GetX(), geom.GetY() ))
