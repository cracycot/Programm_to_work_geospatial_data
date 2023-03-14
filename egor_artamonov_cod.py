import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
# import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep


def open_clean_band(band_path, crop_layer=None):
    """A function that opens a Landsat band as an (rio)xarray object

    Args:
        band_path : list
            A list of paths to the tif files that you wish to combine.

        crop_layer (gpd.GeoDataFrame):
            A geo-dataframe containing the clip extent of interest. NOTE: this will
            fail if the clip extent is in a different CRS than the raster data.

    Returns:
        A single xarray object with the Landsat band data.

    """

    if crop_layer is not None:
        try:
            clip_bound = crop_layer.geometry
            cleaned_band = rxr.open_rasterio(band_path,
                                             masked=True).rio.clip(clip_bound,
                                                                   from_disk=True).squeeze()
        except Exception as err:
            raise AttributeError("Oops, I need a geodataframe object for this to work.")

    else:
        cleaned_band = rxr.open_rasterio(band_path,
                                         masked=True)

    return cleaned_band.squeeze()


def process_bands(paths, crop_layer=None, stack=False):
    """
    Open, clean and crop a list of raster files using rioxarray.

    Args:
        paths (list[str]):
            A list of paths to raster files that could be stacked (of the same
            resolution, crs and spatial extent).

        crop_layer (gpd.GeoDataFrame):
            A geo-dataframe containing the crop geometry that you wish to crop your
            data to.

        stack (boolean):
            If True, return a stacked xarray object. If false will return a list
            of xarray objects.

    Returns:
        Either a list of xarray objects, or a stacked xarray object.
    """

    all_bands = []
    for i, aband in enumerate(paths):
        cleaned = open_clean_band(aband, crop_layer)
        cleaned["band"] = i + 1
        all_bands.append(cleaned)

    if stack:
        return xr.concat(all_bands, dim="band")
    else:
        return all_bands


def plot_xarray_data(xr_data, mode="grayscale", channels=None):
    """A function that plots multiple Landsat bands combined into 1 (rio)xarray object.

    Args:
        xr_data (rxr._io.Dataset | rxr._io.DataArray | list[rxr._io.Dataset]):
            A list of Landsat bands that you wish to plot.

        mode (str):
            A colorspace you wish to visualize your image in. It should be either grayscale, rgb or custom. \
            By default, your image will be created in the grayscale mode.

            * grayscale: Image will be created in shades of gray.
            * rgb: Image will be created in it's natural colors.
            * custom: Image will be created from custom band combination.

        channels (list[int]):
            A list of channels you want to combine.
        save (bool):
            If True, will save generated figure into `results/figures/`. If False, will do nothing.
        title (str):
            A title for figure.

    Returns:
        None
    """
    if mode == "grayscale":
        xr_data.plot.imshow(col="band",
                            col_wrap=3,
                            cmap="Greys_r",
                            title="Grayscale Composite Image",
                            )
    elif mode == "rgb":
        ep.plot_rgb(xr_data.values,
                    rgb=[3, 2, 1],
                    title="RGB Composite Image")
    elif mode == "custom" and channels:
        ep.plot_rgb(xr_data.values,
                    rgb=channels,
                    title="Custom Colored Landsat Image",
                    figsize=(10, 10))
    else:
        raise AttributeError("\'channels\' must be specified when \'mode\' is set to custom.")
    plt.show()


def collect_paths(path_to_folder):
    """A function that collects all paths to Landsat bands in a directory of your choice.

    Args:
        path_to_folder (str):
            A path to folder with Landsat bands.

    Returns:
        (list[str]):
            List of paths to bands' tif files
    """
    # Generate a list of tif files and sort them to ensure bands are in the correct order.
    paths = glob(os.path.join(path_to_folder,
                              "*SR_B*.TIF"))
    if not paths:
        paths = glob(os.path.join(path_to_folder,
                          "*band*.tif"))

    # Sort the data to ensure bands are in the correct order.
    return sorted(paths)


def get_coords_list(band):
    """A function that extracts a

    Args:
        bands (rxr._io.Dataset | rxr._io.DataArray):
            A list of bands you with to get coordinates from.

    Returns:
        A dictionary containing list of coordinates of current band
    """
    coords = band.coords
    return {"x": (coords["x"].values * (10 ** (-4))), "y": coords["y"].values * (10 ** (-5))}

def get_cord_lansat(paths_to_tifs,x,y):
    landsat_post_fire_xr =  process_bands(paths_to_tifs, stack=True)
    coordinates_list = get_coords_list(landsat_post_fire_xr[0])
    return coordinates_list["x"][x], coordinates_list["y"][y]
paths_to_tifs = collect_paths("/Users/kirilllesniak/Downloads/LC09_L2SP_168028_20220321_20220323_02_T1")

landsat_post_fire_xr = process_bands(paths_to_tifs, stack=True)

#plot_xarray_data(landsat_post_fire_xr, "rgb")

coordinates_list = get_coords_list(landsat_post_fire_xr[0])

print(get_cord_lansat(paths_to_tifs, 100,350))