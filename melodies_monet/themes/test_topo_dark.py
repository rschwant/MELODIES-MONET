import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import pycharm_dark as dk

import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(facecolor=dk.facecolor, figsize=(10, 10))

ax = fig.add_subplot(projection=ccrs.SouthPolarStereo())

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
vertices = np.vstack([np.cos(theta), np.sin(theta)]).T
circle = mpath.Path(vertices * radius + center)

ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.gridlines()
ax.set_boundary(circle, transform=ax.transAxes)

plt.tight_layout()

plt.show()
# plt.savefig('test_topo_dark.png')
