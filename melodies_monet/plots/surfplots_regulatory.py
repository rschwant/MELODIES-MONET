#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date: 2018-03-29 10:12:00 -0400 (Thu, 29 Mar 2018) $
# $Revision: 100014 $
# $Author: Barry.Baker@noaa.gov $
# $Id: nemsio2nc4.py 100014 2018-03-29 14:12:00Z Barry.Baker@noaa.gov $
###############################################################

#Original scripts by Patrick Campbell. Adapted to MONET-analysis by Rebecca Schwantes and Barry Baker

import os
import monetio as mio
import monet as monet
import seaborn as sns
#from monet.util.tools import calc_8hr_rolling_max, calc_24hr_ave
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import corrcoef
sns.set_context('paper')
from monet.plots.taylordiagram import TaylorDiagram as td
from matplotlib.colors import ListedColormap
from monet.util.tools import get_epa_region_bounds as get_epa_bounds 
import math
from new_monetio import code_to_move_to_monet as code_m_new
import datetime, pytz
from timezonefinder import TimezoneFinder
from monet.plots.mapgen import draw_map


# from util import write_ncf

def get_localtime(lat,lon,utc_time):
    from datetime import datetime
    from dateutil import tz

    # convert UTC time to local time based on lat/lon
    tf = TimezoneFinder()

    lon2 = lon.values.tolist()
    lat2 = lat.values.tolist()
    timezone_str = tf.timezone_at(lng=lon2, lat=lat2)

    if timezone_str == None:
        if lon > -100.0:
            timezone_str = 'America/New_York'
        else:
            timezone_str = 'America/Los_Angeles'

    #print('Timezone is :', timezone_str)

    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz(timezone_str)
    #print('UTC offset: ', from_zone, to_zone)

    #print('Input UTC time: ', utc_time.values.astype(str))

    utc = datetime.strptime(str(utc_time.values),'%Y-%m-%d %H:%M:%S')
    #print('Reformat UTC: ', utc)

    # Tell the datetime object that it's in UTC time zone since 
    # datetime objects are 'naive' by default
    utc = utc.replace(tzinfo=from_zone)

    # Convert time zone
    dt = utc.astimezone(to_zone)
    local_time = datetime.strptime(str(dt)[:-6],'%Y-%m-%d %H:%M:%S')

    #print('Output local time: ', local_time)

    return local_time

def get_utcoffset(lat,lon):
    from datetime import datetime
    import pytz
    from datetime import datetime, timezone

    # get UTC offset in hour  based on lat/lon
    tf = TimezoneFinder()

    lon2 = lon.values.tolist()
    lat2 = lat.values.tolist()
    timezone_str = tf.timezone_at(lng=lon2, lat=lat2)
    
    if timezone_str == None:
        if lon > -100.0:
            timezone_str = 'America/New_York'
        else:
            timezone_str = 'America/Los_Angeles'
    
    tz = pytz.timezone(timezone_str)
    d=datetime.utcnow()
    uos = tz.utcoffset(d, is_dst=True)
    utchour = uos.seconds/60.0/60.0
    utcday = uos.days

    if utcday < 0:
       utchour = (24-utchour)*-1 # Local - UTC

    return utchour

def make_24hr_regulatory(df, col=None):
    """ Make 24-hour averages """
    #return calc_24hr_ave(df, col)
    return calc_24hr_ave_v1(df, col)

def make_8hr_regulatory(df, col=None):
    """ Make 8-hour rolling average daily """
    #return calc_8hr_rolling_max(df, col, window=8)
    return calc_8hr_rolling_max_v1(df, col, window=8)

def calc_8hr_rolling_max(df, col=None, window=None):
    df.index = df.time_local
    #print(df.index.names)
    df_rolling = df.groupby("siteid")[col].rolling(window, center=True, win_type="boxcar").mean().reset_index().dropna()
    #print('8hr O3: ', df_rolling)
    df_rolling_max = df_rolling.groupby("siteid").resample("D", on="time_local").max().reset_index(drop=True)
    #print('MDA8 O3: ', df_rolling_max)
    df = df.reset_index(drop=True)
    #print(df)
    return df.merge(df_rolling_max, on=["siteid", "time_local"])

def calc_8hr_rolling_max_v1(df, col=None, window=None):
    df.index = df.time_local
    #print(df.index.names)
    df_rolling = df.groupby("siteid")[col].rolling(window, center=True, win_type="boxcar").mean().reset_index().dropna()
    print('8hr O3: ', df_rolling)
    df_rolling_max = df_rolling.groupby("siteid").resample("D", on="time_local").max(min_count=8).reset_index(drop=True).dropna()
    #print('MDA8 O3: ', df_rolling_max)
    df = df.reset_index(drop=True)
    #print(df)
    return df.merge(df_rolling_max, on=["siteid", "time_local"])

def calc_24hr_ave(df, col=None):
    df.index = df.time_local
    df_24hr_ave = df.groupby("siteid")[col].resample("D").mean().reset_index()
    df = df.reset_index(drop=True)
    return df.merge(df_24hr_ave, on=["siteid", "time_local"])

def calc_24hr_ave_v1(df, col=None):
    df.index = df.time_local
    #df_24hr_ave = df.groupby("siteid")[col].resample("D").mean(min_count=8).reset_index(drop=True).dropna()
    df_24hr_ave = (df.groupby("siteid")[col].resample("D").sum(min_count=8)/df.groupby("siteid")[col].resample("D").count()).reset_index().dropna()
    print(df_24hr_ave)
    df = df.reset_index(drop=True)
    return df.merge(df_24hr_ave, on=["siteid", "time_local"])

def make_8hr_regulatory_model(ds, col=None):
    """ Make 8-hour rolling average daily across the model domain"""
    #print(ds[col])

    ds_rolling = ds[col].rolling(time=8, center=True).mean()
    print(ds_rolling)

    #utc_time = ds_rolling.coords['time']
    #lats = ds_rolling.coords['latitude']
    #lons = ds_rolling.coords['longitude']
  
    utc_time = ds_rolling.time
    lats = ds_rolling.latitude
    lons = ds_rolling.longitude

    #print(ds_rolling.dims)
    nt = len(utc_time)
    nx = len(lats[0,:])
    ny = len(lats[:,0])
    print('Model dimensions: ', nt,nx,ny)    

    mda8_o3 = np.zeros((ny,nx))

    start_date = str(utc_time[0].dt.strftime('%Y-%m-%d %H:%M:%S').values)
    print(start_date)
    end_date = str(utc_time[-1].dt.strftime('%Y-%m-%d %H:%M:%S').values)
    print(end_date)

    for i in range(nx):
        for j in range(ny):
            lat = lats[j,i]
            lon = lons[j,i]
            ds_temp = ds_rolling[:,j,i,0]
            #print('Before UTC offset: ', ds_temp)

            df_new = pd.DataFrame()
            df_new[col] = ds_temp.values
            df_new['time_local']=np.zeros(nt)
            
            delh = get_utcoffset(lat,lon)

            dti = pd.date_range(start=start_date,end=end_date,freq='H')
            df_new['time_local']=dti.shift(int(delh),freq='H')
            if (j == 0 and i == 1):
                print('UTC offset at j=0 and j=1: ', delh)
                print(df_new)
            #for t in range(nt):
                #print('UTC time: ', utc_time[t].dt.strftime('%Y-%m-%d %H:%M:%S'))
                #print('UTC time: ',utc_time[t])
            #    df_new['time_local'][t] = get_localtime(lat,lon,utc_time[t].dt.strftime('%Y-%m-%d %H:%M:%S'))
            #print(df_new)

            #ds_temp.coords['time_local'] = local_time
            #ds_temp.assign_coords({'time_local': (lst)})
              
            #ds_new = xr.DataArray(ds_temp.values,coords=lst, dims="time_local")
 
            #ds_new = ds_temp.swap_dims({'time':'time_local'})              
            #ds_new=ds_temp.assign_coords(time=(lst))
            #print('After UTC offset: ', ds_new)

            df_new = df_new.set_index('time_local')
            if (j == ny and i == nx):
                print('After UTC offset: ', df_new)
            mda8_o3[j,i]=df_new[col].resample("D").max().quantile(0.95)
            #print(mda8_o3[0,j,i])
                       
    ds_mean=ds[col].mean(dim='time').squeeze()
    print(ds_mean)
    ds_mean[col+'_MDA8']=(('y', 'x'), mda8_o3)   

    return ds_mean

def make_24hr_regulatory_model(ds, col=None):
    """ Make local 24-hour average across the model domain"""
    print(ds[col])

    utc_time = ds[col].time
    lats = ds[col].latitude
    lons = ds[col].longitude

    nt = len(utc_time)
    nx = len(lats[0,:])
    ny = len(lats[:,0])
    print('Model dimensions: ', nt,nx,ny)

    pm_reg = np.zeros((ny,nx))

    start_date = str(utc_time[0].dt.strftime('%Y-%m-%d %H:%M:%S').values)
    print(start_date)
    end_date = str(utc_time[-1].dt.strftime('%Y-%m-%d %H:%M:%S').values)
    print(end_date)

    for i in range(nx):
        for j in range(ny):
            lat = lats[j,i]
            lon = lons[j,i]
            ds_temp = ds[col][:,j,i,0]
            #print('Before UTC offset: ', ds_temp)

            df_new = pd.DataFrame()
            df_new[col] = ds_temp.values
            df_new['time_local']=np.zeros(nt)

            delh = get_utcoffset(lat,lon)

            dti = pd.date_range(start=start_date,end=end_date,freq='H')
            df_new['time_local']=dti.shift(int(delh),freq='H')
            if (j == 0 and i == 0):
                print('UTC offset at j=0 and i=0: ', delh)
                print(df_new)
            df_new = df_new.set_index('time_local')
            if (j == ny and i == nx):
                print('After UTC offset: ', df_new)
            pm_reg[j,i]=df_new[col].resample("D").mean().mean()

    ds_mean=ds[col].mean(dim='time').squeeze()
    print(ds_mean)
    ds_mean[col+'_24hr']=(('y', 'x'), pm_reg)

    return ds_mean

def calc_default_colors(p_index):
    """ Use default colors """
    x = [dict(color='b', linestyle='--',marker='x'),
         dict(color='g', linestyle='-.',marker='o'),
         dict(color='r', linestyle=':',marker='v'),
         dict(color='c', linestyle='--',marker='^'),
         dict(color='m', linestyle='-.',marker='s')]
    #Repeat these 5 instances over and over if more than 5 lines.
    return x[p_index % 5]

def new_color_map():
    top = mpl.cm.get_cmap('Blues_r', 128)
    bottom = mpl.cm.get_cmap('Oranges', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    return ListedColormap(newcolors, name='OrangeBlue')

def map_projection(f):
    import cartopy.crs as ccrs
    if f.model.lower() == 'cmaq':
        proj = ccrs.LambertConformal(
            central_longitude=f.obj.XCENT, central_latitude=f.obj.YCENT)
    elif f.model.lower() == 'wrfchem' or f.model.lower() == 'rapchem':
        if f.obj.MAP_PROJ == 1:
            proj = ccrs.LambertConformal(
                central_longitude=f.obj.CEN_LON, central_latitude=f.obj.CEN_LAT)
        elif f.MAP_PROJ == 6:
            #Plate Carree is the equirectangular or equidistant cylindrical
            proj = ccrs.PlateCarree(
                central_longitude=f.obj.CEN_LON)
        else:
            raise NotImplementedError('WRFChem projection not supported. Please add to surfplots.py')         
    #Need to add the projections you want to use for the other models here.        
    elif f.model.lower() == 'rrfs':
        proj = ccrs.LambertConformal(
            central_longitude=f.obj.cen_lon, central_latitude=f.obj.cen_lat)
    else: #Let's change this tomorrow to just plot as lambert conformal if nothing provided.
        raise NotImplementedError('Projection not defined for new model. Please add to surfplots.py')
    return proj

def make_spatial_bias(df, column_o=None, label_o=None, column_m=None, 
                      label_m=None, ylabel = None, vdiff=None,
                      outname = 'plot', 
                      regulatory = None,
                      domain_type=None, domain_name=None, fig_dict=None, 
                      text_dict=None,debug=False):
        
    """Creates the MONET-Analysis spatial bias plot."""
    if debug == False:
        plt.ioff()
        
    def_map = dict(states=True,figsize=[10, 5])
    if fig_dict is not None:
        map_kwargs = {**def_map, **fig_dict}
    else:
        map_kwargs = def_map
        
    #If not specified use the PlateCarree projection
    if 'crs' not in map_kwargs:
        map_kwargs['crs'] = ccrs.PlateCarree()
  
    #set default text size
    def_text = dict(fontsize=20)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text
        
    # set ylabel to column if not specified.
    if ylabel is None:
        ylabel = column_o
    
    #Take the mean for each siteid
    df_mean=df.groupby(['siteid'],as_index=False).mean()
       
    #Specify val_max = vdiff. the sp_scatter_bias plot in MONET only uses the val_max value
    #and then uses -1*val_max value for the minimum.
    ax = monet.plots.sp_scatter_bias(
        df_mean, col1=column_o, col2=column_m, map_kwargs=map_kwargs,val_max=vdiff,
        cmap=new_color_map(), edgecolor='k',linewidth=.8)
    
    if domain_type == 'all':
        latmin= 25.0
        lonmin=-130.0
        latmax= 50.0
        lonmax=-60.0
        plt.title(domain_name + ': ' + label_m + ' - ' + label_o,fontweight='bold',**text_kwargs)
    elif domain_type == 'epa_region' and domain_name is not None:
        latmin,lonmin,latmax,lonmax,acro = get_epa_bounds(index=None,acronym=domain_name)
        plt.title('EPA Region ' + domain_name + ': ' + label_m + ' - ' + label_o,fontweight='bold',**text_kwargs)
    else:
        latmin= math.floor(min(df.latitude))
        lonmin= math.floor(min(df.longitude))
        latmax= math.ceil(max(df.latitude))
        lonmax= math.ceil(max(df.longitude))
        plt.title(domain_name + ': ' + label_m + ' - ' + label_o,fontweight='bold',**text_kwargs)
    
    if 'extent' not in map_kwargs:
        map_kwargs['extent'] = [lonmin,lonmax,latmin,latmax]  
    ax.axes.set_extent(map_kwargs['extent'],crs=ccrs.PlateCarree())
    
    #Update colorbar
    f = plt.gcf()
    model_ax = f.get_axes()[0]
    cax = f.get_axes()[1]
    #get the position of the plot axis and use this to rescale nicely the color bar to the height of the plot.
    position_m = model_ax.get_position()
    position_c = cax.get_position()
    cax.set_position([position_c.x0, position_m.y0, position_c.x1 - position_c.x0, (position_m.y1-position_m.y0)*1.1])
    cax.set_ylabel(ylabel,fontweight='bold',**text_kwargs)
    cax.tick_params(labelsize=text_kwargs['fontsize']*0.8,length=10.0,width=2.0,grid_linewidth=2.0)    
    
    #plt.tight_layout(pad=0)
    code_m_new.savefig(outname + '.png',loc=4, height=120, decorate=True, bbox_inches='tight', dpi=200)
    
    if regulatory:
        # calculate hourly data first
        df_copy = df.copy()
        df2 = df_copy.groupby("siteid").resample('H').mean().reset_index()
        print('Hourly data: ',df2)

        #print('After drop: ', df2[column]['000020104'].dropna())
        if ylabel == 'Ozone (ppbv)':
            # calculate MDA8 O3
            column2_o = column_o+'_MDA8'
            column2_m = column_m+'_MDA8'
            df_reg = make_8hr_regulatory(df2, [column_o,column_m]).rename(index=str,columns={column_o+'_y':column2_o,column_m+'_y':column2_m})
            print('After calcualte MDA8O3: ', df_reg)
            # find 95th percentile of MDA8 O3 for each siteid
            df_reg_mean=df_reg.groupby(['siteid'],as_index=False).quantile(.95)

            outname2 = outname+'_MDA8'
            ylabel2 = 'MDA8_O3 (ppbv)'
        elif ylabel == 'PM2.5 (ug/m3)':
            # calcuate 24hr PM2.5
            column2_o = column_o+'_24hr'
            column2_m = column_m+'_24hr'
            df_reg = make_24hr_regulatory(df2, [column_o,column_m]).rename(index=str,columns={column_o+'_y':column2_o,column_m+'_y':column2_m})
            # Take the mean for each siteid
            df_reg_mean=df_reg.groupby(['siteid'],as_index=False).mean()

            outname2 = outname+'_24hr'
            ylabel2 = '24hr_PM2.5 (ug/m3)'

        #Specify val_max = vdiff. the sp_scatter_bias plot in MONET only uses the val_max value
        #and then uses -1*val_max value for the minimum.
        ax2 = monet.plots.sp_scatter_bias(
            df_reg_mean, col1=column2_o, col2=column2_m, map_kwargs=map_kwargs,val_max=vdiff,
            cmap=new_color_map(), edgecolor='k',linewidth=.8)

        if domain_type == 'all':
            latmin= 25.0
            lonmin=-130.0
            latmax= 50.0
            lonmax=-60.0
            plt.title(domain_name + ': ' + label_m + ' - ' + label_o,fontweight='bold',**text_kwargs)
        elif domain_type == 'epa_region' and domain_name is not None:
            latmin,lonmin,latmax,lonmax,acro = get_epa_bounds(index=None,acronym=domain_name)
            plt.title('EPA Region ' + domain_name + ': ' + label_m + ' - ' + label_o,fontweight='bold',**text_kwargs)
        else:
            latmin= math.floor(min(df.latitude))
            lonmin= math.floor(min(df.longitude))
            latmax= math.ceil(max(df.latitude))
            lonmax= math.ceil(max(df.longitude))
            plt.title(domain_name + ': ' + label_m + ' - ' + label_o,fontweight='bold',**text_kwargs)

        if 'extent' not in map_kwargs:
            map_kwargs['extent'] = [lonmin,lonmax,latmin,latmax]
        ax2.axes.set_extent(map_kwargs['extent'],crs=ccrs.PlateCarree())

        #Update colorbar
        f2 = plt.gcf()
        model_ax = f2.get_axes()[0]
        cax = f2.get_axes()[1]
        #get the position of the plot axis and use this to rescale nicely the color bar to the height of the plot.
        position_m = model_ax.get_position()
        position_c = cax.get_position()
        cax.set_position([position_c.x0, position_m.y0, position_c.x1 - position_c.x0, (position_m.y1-position_m.y0)*1.1])
        cax.set_ylabel(ylabel2,fontweight='bold',**text_kwargs)
        cax.tick_params(labelsize=text_kwargs['fontsize']*0.8,length=10.0,width=2.0,grid_linewidth=2.0)

        #plt.tight_layout(pad=0)
        code_m_new.savefig(outname2 + '.png',loc=4, height=120, decorate=True, bbox_inches='tight', dpi=200)

def make_spatial_exceedance(df, column=None, label=None ,
                      ylabel = None, 
                      outname = 'plot',
                      domain_type=None, domain_name=None, fig_dict=None,
                      text_dict=None,debug=False):
    from scipy.stats import scoreatpercentile as score
    from numpy import around

    """Creates the MONET-Analysis spatial bias plot."""
    if debug == False:
        plt.ioff()

    def_map = dict(states=True,figsize=[10, 5])
    if fig_dict is not None:
        map_kwargs = {**def_map, **fig_dict}
    else:
        map_kwargs = def_map

    #If not specified use the PlateCarree projection
    if 'crs' not in map_kwargs:
        map_kwargs['crs'] = ccrs.PlateCarree()

    #set default text size
    def_text = dict(fontsize=20)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text

    # set ylabel to column if not specified.
    if ylabel is None:
        ylabel = column

    #Take the mean for ddeach siteid
    #df_mean=df.groupby(['siteid'],as_index=False).mean()

    # calculate hourly data first
    df_copy = df.copy()
    df2 = df_copy.groupby("siteid").resample('H').mean().reset_index()
    print('Hourly data: ',df2)

    if ylabel == 'Ozone (ppbv)':
        # calculate MDA8 O3
        column2 = column+'_MDA8'
        df_reg = make_8hr_regulatory(df2, column).rename(index=str,columns={column+'_y':column2})
        print(df_reg)
        ylabel2 = 'O3'
        df_reg_mean=df_reg.groupby(["latitude", "longitude"],as_index=False)[column2].quantile(.95)
        print(df_reg_mean)
        df_count = df_reg[df_reg[column2] > 70.].groupby(["latitude", "longitude"],as_index=False)[column2].count()
        print(df_count)
        df_sel = df_reg[df_reg[column2] > 70.].groupby(["latitude", "longitude"],as_index=False)[column2].quantile(.95)
        print(df_sel)
    elif ylabel == 'PM2.5 (ug/m3)':
        # calcuate 24hr PM2.5
        column2 = column+'_24hr'
        df_reg = make_24hr_regulatory(df2, column).rename(index=str,columns={column+'_y':column2})
        print(df_reg)
        ylabel2 = 'PM2.5'
        df_reg_mean=df_reg.groupby(["latitude", "longitude"],as_index=False)[column2].mean()
        print(df_reg_mean)
        df_count = df_reg[df_reg[column2] > 35.].groupby(["latitude", "longitude"],as_index=False)[column2].count()
        print(df_count)
        df_sel = df_reg[df_reg[column2] > 35.].groupby(["latitude", "longitude"],as_index=False)[column2].mean()
        print(df_sel)

    if not df_count.empty:
        ax = draw_map(**map_kwargs)

        dfnew = df_count[["latitude", "longitude", column2]].copy(deep=True)
        print(dfnew)
        dfnew["nexceday"]=df_count[column2]
        dfnew[column2]=df_sel[column2]
        print(dfnew)    

        dfnew[['latitude','longitude','nexceday',column2]].to_csv(outname+'_exceendance.csv', index=False)

        top = score(dfnew["nexceday"].abs(), per=95)
        print(top)
        val_max = 15
        if val_max is not None:
            top = val_max
        x, y = dfnew.longitude.values, dfnew.latitude.values
        dfnew["nexceday_size"] = dfnew["nexceday"].abs() / top * 100.0
        dfnew.loc[dfnew["nexceday_size"] > 300, "nexceday_size"] = 300.0
        print(dfnew)

        cmap = plt.get_cmap('YlOrRd')
        dfnew.plot.scatter(
            x="longitude", y="latitude", c=dfnew["nexceday"], s=dfnew["nexceday_size"], vmin=0, vmax=top, ax=ax, colorbar=True,
            cmap=cmap, edgecolor='k',linewidth=.8)
        
        if domain_type == 'all':
            latmin= 25.0
            lonmin=-130.0
            latmax= 50.0
            lonmax=-60.0
            plt.title(domain_name + ': ' + label, fontweight='bold',**text_kwargs)
        elif domain_type == 'epa_region' and domain_name is not None:
            latmin,lonmin,latmax,lonmax,acro = get_epa_bounds(index=None,acronym=domain_name)
            plt.title('EPA Region ' + domain_name + ': ' + label, fontweight='bold',**text_kwargs)
        else:
            latmin= math.floor(min(df.latitude))
            lonmin= math.floor(min(df.longitude))
            latmax= math.ceil(max(df.latitude))
            lonmax= math.ceil(max(df.longitude))
            plt.title(domain_name + ': ' + label, fontweight='bold',**text_kwargs)

        if 'extent' not in map_kwargs:
            map_kwargs['extent'] = [lonmin,lonmax,latmin,latmax]
        ax.axes.set_extent(map_kwargs['extent'],crs=ccrs.PlateCarree())

        #Update colorbar
        f = plt.gcf()
        model_ax = f.get_axes()[0]
        cax = f.get_axes()[1]
        #get the position of the plot axis and use this to rescale nicely the color bar to the height of the plot.
        position_m = model_ax.get_position()
        position_c = cax.get_position()
        cax.set_position([position_c.x0, position_m.y0, position_c.x1 - position_c.x0, (position_m.y1-position_m.y0)*1.1])
        cax.set_ylabel(ylabel2+' exceedance days',fontweight='bold',**text_kwargs)
        cax.tick_params(labelsize=text_kwargs['fontsize']*0.8,length=10.0,width=2.0,grid_linewidth=2.0)

        #plt.tight_layout(pad=0)
        code_m_new.savefig(outname + '_exceendance.png',loc=4, height=120, decorate=True, bbox_inches='tight', dpi=200)

def make_timeseries(df, column=None, label=None, fig=None, ax=None, avg_window=None, ylabel=None,
                    vmin = None, vmax = None,
                    domain_type=None, domain_name=None,
                    plot_dict=None, fig_dict=None, text_dict=None,debug=False):
    """Creates the MONET-Analysis time series plot."""
    if debug == False:
        plt.ioff()
    #First define items for all plots
    #set default text size
    def_text = dict(fontsize=14)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text
    # set ylabel to column if not specified.
    if ylabel is None:
        ylabel = column
    if label is not None:
        plot_dict['label'] = label
    if vmin is not None and vmax is not None:
        plot_dict['ylim'] = [vmin,vmax]
    #scale the fontsize for the x and y labels by the text_kwargs
    plot_dict['fontsize'] = text_kwargs['fontsize']*0.8
    
    #Then, if no plot has been created yet, create a plot and plot the obs.
    if ax is None: 
        #First define the colors for the observations.
        obs_dict = dict(color='k', linestyle='-',marker='*', linewidth=1.2, markersize=6.)
        if plot_dict is not None:
            #Whatever is not defined in the yaml file is filled in with the obs_dict here.
            plot_kwargs = {**obs_dict, **plot_dict}
        else:
            plot_kwargs = obs_dict
        # create the figure
        if fig_dict is not None:
            fig,ax = plt.subplots(**fig_dict)    
        else: 
            fig,ax = plt.subplots(figsize=(10,6))
        # plot the line
        if avg_window is None:
            ax = df[column].plot(ax=ax, **plot_kwargs)
        else:
            ax = df[column].resample(avg_window).mean().plot(ax=ax, legend=True, **plot_kwargs)
    
    # If plot has been created add to the current axes.
    else:
        # this means that an axis handle already exists and use it to plot the model output.
        if avg_window is None:
            ax = df[column].plot(ax=ax, legend=True, **plot_dict)
        else:
            ax = df[column].resample(avg_window).mean().plot(ax=ax, legend=True, **plot_dict)    

    #Set parameters for all plots
    ax.set_ylabel(ylabel,fontweight='bold',**text_kwargs)
    ax.set_xlabel(df.index.name,fontweight='bold',**text_kwargs)
    ax.legend(frameon=False,fontsize=text_kwargs['fontsize']*0.8)
    ax.tick_params(axis='both',length=10.0,direction='inout')
    ax.tick_params(axis='both',which='minor',length=5.0,direction='out')
    ax.legend(frameon=False,fontsize=text_kwargs['fontsize']*0.8,
              bbox_to_anchor=(1.0, 0.9), loc='center left')
    if domain_type is not None and domain_name is not None:
        if domain_type == 'epa_region':
            ax.set_title('EPA Region ' + domain_name,fontweight='bold',**text_kwargs)
        else:
            ax.set_title(domain_name,fontweight='bold',**text_kwargs)
    return fig,ax

def make_timeseries_regulatory(df, column=None, label=None, fig2=None, ax2=None, avg_window=None, ylabel=None,
                    vmin = None, vmax = None,
                    domain_type=None, domain_name=None,
                    plot_dict=None, fig_dict=None, text_dict=None,debug=False):
    """Creates the MONET-Analysis time series plot."""
    if debug == False:
        plt.ioff()
    #First define items for all plots
    #set default text size
    def_text = dict(fontsize=14)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text
    # set ylabel to column if not specified.
    if ylabel is None:
        ylabel = column
    if label is not None:
        plot_dict['label'] = label
    if vmin is not None and vmax is not None:
        plot_dict['ylim'] = [vmin,vmax]
    #scale the fontsize for the x and y labels by the text_kwargs
    plot_dict['fontsize'] = text_kwargs['fontsize']*0.8

    # calculate hourly data first
    df_copy = df.copy()
    df2 = df_copy.groupby("siteid").resample('H').mean().reset_index()
    print('Hourly data: ',df2)

    if ylabel == 'Ozone (ppbv)':
        # calculate MDA8 O3
        column2 = column+'_MDA8'
        df_reg = make_8hr_regulatory(df2, column).rename(index=str,columns={column+'_y':column2})
        print('After calcualte MDA8O3: ', df_reg)
        ylabel2 = 'MDA8_O3 (ppbv)'
        df_reg.index=df_reg.time_local
        df_reg_avg = df_reg[column2].resample('D').mean()

    elif ylabel == 'PM2.5 (ug/m3)':
        # calcuate 24hr PM2.5
        column2 = column+'_24hr'
        df_reg = make_24hr_regulatory(df2, column).rename(index=str,columns={column+'_y':column2})
        ylabel2 = '24hr_PM2.5 (ug/m3)'
        df_reg.index=df_reg.time_local
        df_reg_avg = df_reg[column2].resample('D').mean()

    #Then, if no plot has been created yet, create a plot and plot the obs.
    if ax2 is None:
        #First define the colors for the observations.
        obs_dict = dict(color='k', linestyle='-',marker='*', linewidth=1.2, markersize=6.)
        if plot_dict is not None:
            #Whatever is not defined in the yaml file is filled in with the obs_dict here.
            plot_kwargs = {**obs_dict, **plot_dict}
        else:
            plot_kwargs = obs_dict
        # create the figure
        if fig_dict is not None:
            fig2,ax2 = plt.subplots(**fig_dict)
        else:
            fig2,ax2 = plt.subplots(figsize=(10,6))
        # plot the line

        ax2 = df_reg_avg.plot(ax=ax2, legend=True, **plot_kwargs)
        print('Diag ts:', df[column].resample(avg_window).mean())

    # If plot has been created add to the current axes.
    else:
        # this means that an axis handle already exists and use it to plot the model output.
        ax2 = df_reg_avg.plot(ax=ax2, legend=True, **plot_dict)
        print('Diag ts:', df[column].resample(avg_window).mean())

    #Set parameters for all plots
    ax2.set_ylabel(ylabel2,fontweight='bold',**text_kwargs)
    ax2.set_xlabel(df.index.name,fontweight='bold',**text_kwargs)
    ax2.legend(frameon=False,fontsize=text_kwargs['fontsize']*0.8)
    ax2.tick_params(axis='both',length=10.0,direction='inout')
    ax2.tick_params(axis='both',which='minor',length=5.0,direction='out')
    ax2.legend(frameon=False,fontsize=text_kwargs['fontsize']*0.8,
              bbox_to_anchor=(1.0, 0.9), loc='center left')
    if domain_type is not None and domain_name is not None:
        if domain_type == 'epa_region':
            ax2.set_title('EPA Region ' + domain_name,fontweight='bold',**text_kwargs)

        else:
            ax2.set_title(domain_name,fontweight='bold',**text_kwargs)
    return fig2,ax2

def make_taylor(df, column_o=None, label_o='Obs', column_m=None, label_m='Model',
                tyf=None,
                dia=None, ylabel=None, ty_scale=1.5,
                domain_type=None, domain_name=None,
                plot_dict=None, fig_dict=None, text_dict=None,debug=False):
    """Creates the MONET-Analysis taylor plot."""
    #First define items for all plots
    if debug == False:
        plt.ioff()

    #set default text size
    def_text = dict(fontsize=14.0)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text
    # set ylabel to column if not specified.
    if ylabel is None:
        ylabel = column_o
    #Then, if no plot has been created yet, create a plot and plot the first pair.
    if dia is None:
        # create the figure
        if fig_dict is not None:
            tyf = plt.figure(**fig_dict)
        else:
            tyf = plt.figure(figsize=(12,10))
        sns.set_style('ticks')
        # plot the line
        dia = td(df[column_o].std(), scale=ty_scale, fig=tyf,
                               rect=111, label=label_o)
        plt.grid(linewidth=1, alpha=.5)
        cc = corrcoef(df[column_o].values, df[column_m].values)[0, 1]
        dia.add_sample(df[column_m].std(), cc, zorder=9, label=label_m, **plot_dict)
    # If plot has been created add to the current axes.
    else:
        # this means that an axis handle already exists and use it to plot another model
        cc = corrcoef(df[column_o].values, df[column_m].values)[0, 1]
        dia.add_sample(df[column_m].std(), cc, zorder=9, label=label_m, **plot_dict)
    #Set parameters for all plots
    contours = dia.add_contours(colors='0.5')
    plt.clabel(contours, inline=1, fontsize=text_kwargs['fontsize']*0.8)
    plt.grid(alpha=.5)
    plt.legend(frameon=False,fontsize=text_kwargs['fontsize']*0.8,
               bbox_to_anchor=(0.75, 0.93), loc='center left')
    if domain_type is not None and domain_name is not None:
        if domain_type == 'epa_region':
            plt.title('EPA Region ' + domain_name,fontweight='bold',**text_kwargs)
        else:
            plt.title(domain_name,fontweight='bold',**text_kwargs)
    ax = plt.gca()
    ax.axis["left"].label.set_text('Standard Deviation: '+ylabel)
    ax.axis["top"].label.set_text('Correlation')
    ax.axis["left"].label.set_fontsize(text_kwargs['fontsize'])
    ax.axis["top"].label.set_fontsize(text_kwargs['fontsize'])
    ax.axis["left"].label.set_fontweight('bold')
    ax.axis["top"].label.set_fontweight('bold')
    ax.axis["top"].major_ticklabels.set_fontsize(text_kwargs['fontsize']*0.8)
    ax.axis["left"].major_ticklabels.set_fontsize(text_kwargs['fontsize']*0.8)
    ax.axis["right"].major_ticklabels.set_fontsize(text_kwargs['fontsize']*0.8)
    return tyf,dia


def make_taylor_regulatory(df, column_o=None, label_o='Obs', column_m=None, label_m='Model', 
                tyf2=None,  
                dia2=None,ylabel=None, ty_scale=1.5,
                domain_type=None, domain_name=None,
                plot_dict=None, fig_dict=None, text_dict=None,debug=False):
    """Creates the MONET-Analysis taylor plot."""
    #First define items for all plots
    if debug == False:
        plt.ioff()
        
    #set default text size
    def_text = dict(fontsize=14.0)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text
    # set ylabel to column if not specified.
    if ylabel is None:
        ylabel = column_o

    # calculate hourly data first
    df_copy = df.copy()
    df2 = df_copy.groupby("siteid").resample('H').mean().reset_index()

    if ylabel == 'Ozone (ppbv)':
        # calculate MDA8 O3
        column2_o = column_o+'_MDA8'
        column2_m = column_m+'_MDA8'
        df_reg = make_8hr_regulatory(df2, [column_o,column_m]).rename(index=str,columns={column_o+'_y':column2_o,column_m+'_y':column2_m})
        ylabel2 = 'MDA8_O3 (ppbv)'
    elif ylabel == 'PM2.5 (ug/m3)':
        # calcuate 24hr PM2.5
        column2_o = column_o+'_24hr'
        column2_m = column_m+'_24hr'
        df_reg = make_24hr_regulatory(df2, [column_o,column_m]).rename(index=str,columns={column_o+'_y':column2_o,column_m+'_y':column2_m})
        ylabel2 = '24hr_PM2.5 (ug/m3)'

    if dia2 is None:
        # create the figure
        if fig_dict is not None:
            tyf2 = plt.figure(**fig_dict)
        else:
            tyf2 = plt.figure(figsize=(12,10))
        sns.set_style('ticks')

        # plot the line
        dia2 = td(df_reg[column2_o].std(), scale=ty_scale, fig=tyf2,
                           rect=111, label=label_o)
        plt.grid(linewidth=1, alpha=.5)
        cc = corrcoef(df_reg[column2_o].values, df_reg[column2_m].values)[0, 1]
        dia2.add_sample(df_reg[column2_m].std(), cc, zorder=9, label=label_m, **plot_dict)

    # If plot has been created add to the current axes.
    else:
        # this means that an axis handle already exists and use it to plot another model
        cc = corrcoef(df_reg[column2_o].values, df_reg[column2_m].values)[0, 1]
        dia2.add_sample(df_reg[column2_m].std(), cc, zorder=9, label=label_m, **plot_dict)

    #Set parameters for all plots
    contours = dia2.add_contours(colors='0.5')
    plt.clabel(contours, inline=1, fontsize=text_kwargs['fontsize']*0.8)
    plt.grid(alpha=.5)
    plt.legend(frameon=False,fontsize=text_kwargs['fontsize']*0.8,
           bbox_to_anchor=(0.75, 0.93), loc='center left')
    if domain_type is not None and domain_name is not None:
        if domain_type == 'epa_region':
            plt.title('EPA Region ' + domain_name,fontweight='bold',**text_kwargs)
        else:
            plt.title(domain_name,fontweight='bold',**text_kwargs)
    ax = plt.gca()
    ax.axis["left"].label.set_text('Standard Deviation: '+ylabel2)
    ax.axis["top"].label.set_text('Correlation')
    ax.axis["left"].label.set_fontsize(text_kwargs['fontsize'])
    ax.axis["top"].label.set_fontsize(text_kwargs['fontsize'])
    ax.axis["left"].label.set_fontweight('bold')
    ax.axis["top"].label.set_fontweight('bold')
    ax.axis["top"].major_ticklabels.set_fontsize(text_kwargs['fontsize']*0.8)
    ax.axis["left"].major_ticklabels.set_fontsize(text_kwargs['fontsize']*0.8)
    ax.axis["right"].major_ticklabels.set_fontsize(text_kwargs['fontsize']*0.8)

    return tyf2,dia2

def make_spatial_overlay(df, vmodel, column_o=None, label_o=None, column_m=None, 
                      label_m=None, ylabel = None, vmin=None, vmax=None,
                      nlevels = None, proj = None, outname = 'plot',
                      regulatory = None, 
                      domain_type=None, domain_name=None, fig_dict=None, 
                      text_dict=None,debug=False):
        
    """Creates the MONET-Analysis spatial overlay plot."""
    if debug == False:
        plt.ioff()
        
    def_map = dict(states=True,figsize=[15, 8])
    if fig_dict is not None:
        map_kwargs = {**def_map, **fig_dict}
    else:
        map_kwargs = def_map
  
    #set default text size
    def_text = dict(fontsize=20)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text
        
    # set ylabel to column if not specified.
    if ylabel is None:
        ylabel = column_o
    
    #Take the mean for each siteid
    df_mean=df.groupby(['siteid'],as_index=False).mean()
    
    #Take the mean over time for the model output
    vmodel_mean = vmodel[column_m].mean(dim='time').squeeze()
    
    #Determine the domain
    if domain_type == 'all':
        latmin= 25.0
        lonmin=-130.0
        latmax= 50.0
        lonmax=-60.0
        title_add = domain_name + ': '
    elif domain_type == 'epa_region' and domain_name is not None:
        latmin,lonmin,latmax,lonmax,acro = get_epa_bounds(index=None,acronym=domain_name)
        title_add = 'EPA Region ' + domain_name + ': '
    else:
        latmin= math.floor(min(df.latitude))
        lonmin= math.floor(min(df.longitude))
        latmax= math.ceil(max(df.latitude))
        lonmax= math.ceil(max(df.longitude))
        title_add = domain_name + ': '
    
    #Map the model output first.
    cbar_kwargs = dict(aspect=15,shrink=.8)
    
    #Add options that this could be included in the fig_kwargs in yaml file too.
    if 'extent' not in map_kwargs:
        map_kwargs['extent'] = [lonmin,lonmax,latmin,latmax] 
    if 'crs' not in map_kwargs:
        map_kwargs['crs'] = proj
    
    #With pcolormesh, a Warning shows because nearest interpolation may not work for non-monotonically increasing regions.
    #Because I do not want to pull in the edges of the lat lon for every model I switch to contourf.
    #First determine colorbar, so can use the same for both contourf and scatter
    if vmin == None and vmax == None:
        vmin = np.min((vmodel_mean.quantile(0.01), df_mean[column_o].quantile(0.01)))
        vmax = np.max((vmodel_mean.quantile(0.99), df_mean[column_o].quantile(0.99)))
        
    if nlevels == None:
        nlevels = 21
    
    clevel = np.linspace(vmin,vmax,nlevels)
    cmap = mpl.cm.get_cmap('Spectral_r',nlevels-1) 
    norm = mpl.colors.BoundaryNorm(clevel, ncolors=cmap.N, clip=False)
        
    #I add extend='both' here because the colorbar is setup to plot the values outside the range
    ax = vmodel_mean.monet.quick_contourf(cbar_kwargs=cbar_kwargs, figsize=map_kwargs['figsize'], map_kws=map_kwargs,
                                robust=True, norm=norm, cmap=cmap, levels=clevel, extend='both') 
    
    plt.gcf().canvas.draw() 
    plt.tight_layout(pad=0)
    plt.title(title_add + label_o + ' overlaid on ' + label_m,fontweight='bold',**text_kwargs)
     
    ax.axes.scatter(df_mean.longitude.values, df_mean.latitude.values,s=30,c=df_mean[column_o], 
                    transform=ccrs.PlateCarree(), edgecolor='b', linewidth=.50, norm=norm, 
                    cmap=cmap)
    ax.axes.set_extent(map_kwargs['extent'],crs=ccrs.PlateCarree())    
    
    #Uncomment these lines if you update above just to verify colorbars are identical.
    #Also specify plot above scatter = ax.axes.scatter etc.
    #cbar = ax.figure.get_axes()[1] 
    #plt.colorbar(scatter,ax=ax)
    
    #Update colorbar
    f = plt.gcf()
    model_ax = f.get_axes()[0]
    cax = f.get_axes()[1]
    #get the position of the plot axis and use this to rescale nicely the color bar to the height of the plot.
    position_m = model_ax.get_position()
    position_c = cax.get_position()
    cax.set_position([position_c.x0, position_m.y0, position_c.x1 - position_c.x0, (position_m.y1-position_m.y0)*1.1])
    cax.set_ylabel(ylabel,fontweight='bold',**text_kwargs)
    cax.tick_params(labelsize=text_kwargs['fontsize']*0.8,length=10.0,width=2.0,grid_linewidth=2.0)    
    
    #plt.tight_layout(pad=0)
    code_m_new.savefig(outname + '.png',loc=4, height=100, decorate=True, bbox_inches='tight', dpi=150)

    if regulatory:
        # calculate hourly data first
        df_copy = df.copy()
        df2 = df_copy.groupby("siteid").resample('H').mean().reset_index()
        print('Hourly data: ',df2)

        #print('After drop: ', df2[column]['000020104'].dropna())
        if ylabel == 'Ozone (ppbv)':
            # calculate MDA8 O3
            column2_o = column_o+'_MDA8'
            column2_m = column_m+'_MDA8'
            df_reg = make_8hr_regulatory(df2, [column_o,column_m]).rename(index=str,columns={column_o+'_y':column2_o,column_m+'_y':column2_m})
            print('After calcualte MDA8O3: ', df_reg)
            # find 95th percentile of MDA8 O3 for each siteid
            df_reg_mean=df_reg.groupby(['siteid'],as_index=False).quantile(.95)

            # calculate MDA8 O3 for the model output
            #vmodel2 = vmodel[column_m].rolling(time=8, center=True).mean()
            #vmodel_reg = vmodel2.resample("D").max()

            vmodel_reg_mean = make_8hr_regulatory_model(vmodel, column_m)
            print('Model regulatory values: ',vmodel_reg_mean)

            outname2 = outname+'_MDA8'
            ylabel2 = 'MDA8_O3 (ppbv)'
        elif ylabel == 'PM2.5 (ug/m3)':
            # calcuate 24hr PM2.5
            column2_o = column_o+'_24hr'
            column2_m = column_m+'_24hr'
            df_reg = make_24hr_regulatory(df2, [column_o,column_m]).rename(index=str,columns={column_o+'_y':column2_o,column_m+'_y':column2_m})
            # Take the mean for each siteid
            df_reg_mean=df_reg.groupby(['siteid'],as_index=False).mean()

            # calculate 24hr PM2.5 for the model output
            #vmodel_reg = vmodel[column_m].resample(time='1D').mean()
            #vmodel_reg_mean = vmodel_reg.mean(dim='time').squeeze()
 
            vmodel_reg_mean = make_24hr_regulatory_model(vmodel, column_m)
            print('Model regulatory values: ',vmodel_reg_mean)

            outname2 = outname+'_24hr'
            ylabel2 = '24hr_PM2.5 (ug/m3)'

        #Determine the domain
        if domain_type == 'all':
            latmin= 25.0
            lonmin=-130.0
            latmax= 50.0
            lonmax=-60.0
            title_add = domain_name + ': '
        elif domain_type == 'epa_region' and domain_name is not None:
            latmin,lonmin,latmax,lonmax,acro = get_epa_bounds(index=None,acronym=domain_name)
            title_add = 'EPA Region ' + domain_name + ': '
        else:
            latmin= math.floor(min(df.latitude))
            lonmin= math.floor(min(df.longitude))
            latmax= math.ceil(max(df.latitude))
            lonmax= math.ceil(max(df.longitude))
            title_add = domain_name + ': '

        #Map the model output first.
        cbar_kwargs = dict(aspect=15,shrink=.8)

        #Add options that this could be included in the fig_kwargs in yaml file too.
        if 'extent' not in map_kwargs:
            map_kwargs['extent'] = [lonmin,lonmax,latmin,latmax]
        if 'crs' not in map_kwargs:
            map_kwargs['crs'] = proj

        #With pcolormesh, a Warning shows because nearest interpolation may not work for non-monotonically increasing regions.
        #Because I do not want to pull in the edges of the lat lon for every model I switch to contourf.
        #First determine colorbar, so can use the same for both contourf and scatter
        if ylabel2 == 'MDA8_O3 (ppbv)':
            pmin = 30.
            pmax = 70.
        else:
            pmin = vmin
            pmax = vmax

        if nlevels == None:
            nlevels = 21

        clevel = np.linspace(pmin,pmax,nlevels)
        cmap = mpl.cm.get_cmap('Spectral_r',nlevels-1)
        norm = mpl.colors.BoundaryNorm(clevel, ncolors=cmap.N, clip=False)

        #I add extend='both' here because the colorbar is setup to plot the values outside the range
        ax2 = vmodel_reg_mean.monet.quick_contourf(cbar_kwargs=cbar_kwargs, figsize=map_kwargs['figsize'], map_kws=map_kwargs,
                                robust=True, norm=norm, cmap=cmap, levels=clevel, extend='both')

        plt.gcf().canvas.draw()
        plt.tight_layout(pad=0)
        plt.title(title_add + label_o + ' overlaid on ' + label_m,fontweight='bold',**text_kwargs)

        ax2.axes.scatter(df_reg_mean.longitude.values, df_reg_mean.latitude.values,s=30,c=df_reg_mean[column2_o],
                    transform=ccrs.PlateCarree(), edgecolor='b', linewidth=.50, norm=norm,
                    cmap=cmap)
        ax2.axes.set_extent(map_kwargs['extent'],crs=ccrs.PlateCarree())

        #Uncomment these lines if you update above just to verify colorbars are identical.
        #Also specify plot above scatter = ax.axes.scatter etc.
        #cbar = ax.figure.get_axes()[1] 
        #plt.colorbar(scatter,ax=ax)

        #Update colorbar
        f2 = plt.gcf()
        model_ax = f2.get_axes()[0]
        cax = f2.get_axes()[1]
        #get the position of the plot axis and use this to rescale nicely the color bar to the height of the plot.
        position_m = model_ax.get_position()
        position_c = cax.get_position()
        cax.set_position([position_c.x0, position_m.y0, position_c.x1 - position_c.x0, (position_m.y1-position_m.y0)*1.1])
        cax.set_ylabel(ylabel2,fontweight='bold',**text_kwargs)
        cax.tick_params(labelsize=text_kwargs['fontsize']*0.8,length=10.0,width=2.0,grid_linewidth=2.0)

        #plt.tight_layout(pad=0)
        code_m_new.savefig(outname2 + '.png',loc=4, height=100, decorate=True, bbox_inches='tight', dpi=150)

def calculate_boxplot(df, column=None, label=None, plot_dict=None, comb_bx = None, label_bx = None):
    if comb_bx is None and label_bx is None:
        comb_bx = pd.DataFrame()
        label_bx = []
        #First define the colors for the observations.
        obs_dict = dict(color='gray', linestyle='-',marker='x', linewidth=1.2, markersize=6.)
        if plot_dict is not None:
            #Whatever is not defined in the yaml file is filled in with the obs_dict here.
            plot_kwargs = {**obs_dict, **plot_dict}
        else:
            plot_kwargs = obs_dict
    else:
        plot_kwargs = plot_dict
    #For all, a column to the dataframe and append the label info to the list.
    plot_kwargs['column'] = column
    plot_kwargs['label'] = label
    comb_bx[label] = df[column]
    label_bx.append(plot_kwargs)
    
    return comb_bx, label_bx
    
def make_boxplot(comb_bx, label_bx, ylabel = None, vmin = None, vmax = None, outname='plot',
                 regulatory=None,
                 domain_type=None, domain_name=None,
                 plot_dict=None, fig_dict=None,text_dict=None,debug=False):
    
    """Creates the MONET-Analysis box plot. """
    if debug == False:
        plt.ioff()
    #First define items for all plots
    #set default text size
    def_text = dict(fontsize=14)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text
    # set ylabel to column if not specified.
    if ylabel is None:
        ylabel = label_bx[0][column]
    
    if regulatory:
        if ylabel == 'Ozone (ppbv)':
            outname = outname+'_MDA8'
            ylabel2 = 'MDA8_O3 (ppbv)'
        elif ylabel == 'PM2.5 (ug/m3)':
            outname = outname+'_24hr'
            ylabel2 = '24hr_PM2.5 (ug/m3)'

    #Fix the order and palate colors
    order_box = []
    pal = {}
    for i in range(len(label_bx)):
        order_box.append(label_bx[i]['label'])
        pal[label_bx[i]['label']] = label_bx[i]['color']
        
    #Make plot
    if fig_dict is not None:
        f,ax = plt.subplots(**fig_dict)    
    else: 
        f,ax = plt.subplots(figsize=(8,8))
    #Define characteristics of boxplot.
    boxprops = {'edgecolor': 'k', 'linewidth': 1.5}
    lineprops = {'color': 'k', 'linewidth': 1.5}
    boxplot_kwargs = {'boxprops': boxprops, 'medianprops': lineprops,
                  'whiskerprops': lineprops, 'capprops': lineprops,
                  'fliersize' : 2.0, 
                  'flierprops': dict(marker='*', 
                                     markerfacecolor='blue', 
                                     markeredgecolor='none',
                                     markersize = 6.0),
                  'width': 0.75, 'palette': pal,
                  'order': order_box,
                  'showmeans': True, 
                  'meanprops': {'marker': ".", 'markerfacecolor': 'black', 
                                'markeredgecolor': 'black',
                               'markersize': 20.0}}
    sns.set_style("whitegrid")
    sns.set_style("ticks")
    sns.boxplot(ax=ax,x="variable", y="value",data=pd.melt(comb_bx), **boxplot_kwargs)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel,fontweight='bold',**text_kwargs)
    if regulatory:
        ax.set_ylabel(ylabel2,fontweight='bold',**text_kwargs)
    ax.tick_params(labelsize=text_kwargs['fontsize']*0.8)
    if domain_type is not None and domain_name is not None:
        if domain_type == 'epa_region':
            ax.set_title('EPA Region ' + domain_name,fontweight='bold',**text_kwargs)
        else:
            ax.set_title(domain_name,fontweight='bold',**text_kwargs)
    if vmin is not None and vmax is not None:
        ax.set_ylim(ymin = vmin, ymax = vmax)
    
    plt.tight_layout()
    code_m_new.savefig(outname + '.png',loc=4, height=100, decorate=True, bbox_inches='tight', dpi=200)
    
    print(order_box)

