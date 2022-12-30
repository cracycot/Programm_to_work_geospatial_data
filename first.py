from pyhdf import HDF
from pyhdf.SD import SD,SDC
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
import h5py
import numpy as np

#arr = np.random.rand(10000)

def get_rastr(way):
    gdalData = gdal.Open(way)
    raster = gdalData.ReadAsArray()
    mas = np.array(raster)
    return mas

def open_file(way):
    return SD(way,SDC.WRITE|SDC.CREATE|SDC.READ)

def select(file,name):
    return file.select(name)
    print("hello_world")

def main():
    ways = {"mod14" : "/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD14.hdf",
    "mod3" : "/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD03.hdf" ,
    "mod2" : "/Users/kirilllesniak/Downloads/20210314_113600_AQUA_MOD021KM.hdf"}
    file = open_file(way=ways["mod2"])
    data = select(file, "Band_250M")
    gdalData = gdal.Open(ways["mod2"])
    print("Geo transform", gdalData.GetGeoTransform())
    print(get_rastr(ways["mod2"]))
    print(data.get())
    #print(getattr(file , "Band_250M"))
    #file.get()
    #print(file.get())

if __name__ == '__main__':
    main()
