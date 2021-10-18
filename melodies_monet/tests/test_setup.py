import os
import sys
import argparse
import math
import numpy as np
import scipy as sp
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,
    default=1,
    help='random generator seed')
parser.add_argument('--nlat', type=int,
    default=180)
parser.add_argument('--nlon', type=int,
    default=360)
parser.add_argument('--l_max', type=int,
    default=2)
parser.add_argument('--start', type=str,
    default='20200801',
    help='start date (yyyymmdd)')
parser.add_argument('--end', type=str,
    default='20200831',
    help='end date (yyyymmdd)')
args = parser.parse_args()

"""
Generate uniform grid
"""
nlat, nlon = args.nlat, args.nlon
lat_edges = np.linspace(-90, 90, nlat+1, endpoint=True, dtype=float)
lat = 0.5 * (lat_edges[0:nlat] + lat_edges[1:nlat+1])
lat_min, lat_max = lat_edges[0:nlat], lat_edges[1:nlat+1]
deg_to_rad = math.pi / 180.0
weight = np.abs(np.sin(deg_to_rad * lat_max) - np.sin(deg_to_rad * lat_min))
lon_edges = np.linspace(-180, 180, nlon+1, endpoint=True, dtype=float)
lon = 0.5 * (lon_edges[0:nlon] + lon_edges[1:nlon+1])
