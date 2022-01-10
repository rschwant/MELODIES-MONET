import cartopy.feature as cfeature
from cartopy.io import shapereader

resolution = '110m'
category = 'cultural'
name = 'admin_0_countries'

shapefile = shapereader.natural_earth(resolution, category, name)
reader = shapereader.Reader(shapefile)
countries = reader.records()
for country in countries:
    print(country.attributes['SOVEREIGNT'])
    print(country.geometry)
