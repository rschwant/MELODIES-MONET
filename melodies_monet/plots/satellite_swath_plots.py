import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs

sns.set_context('paper')


def make_swath(ds):
    for granule in ds:
        print(granule)
