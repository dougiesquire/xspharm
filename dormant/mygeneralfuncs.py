"""
    General functions used regularly when processing CAFE data - need to organise these better once library grows
    Author: Dougie Squire
    Date created: 09/03/2018
    Python Version: 3.5
"""

__all__ = ['timer','gen_monyear','make_lon_positive','convert_to_lower_freq','calc_boxavg_latlon','make_landsea_mask']

from scipy import interpolate
import xarray as xr
import numpy as np
import time

class timer(object):
    '''
        File name: timer

        Description: class for timing code snippets 

        Author: Dougie Squire
        Date created: 21/03/2018
        Python Version: 3.5

        Usage:
            with timer():
                # do something
    '''
    
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('   f{self.name}')
        print(f'   Elapsed: {time.time() - self.tstart} sec')
        
def gen_monyear(st_date, nd_date):
    '''
        File name: gen_monyear

        Description: Generator for looping over months and years

        Author: Dougie Squire
        Date created: 08/02/2018
        Python Version: 3.5

        Args:
            st_date (:class:`tuple`): Tuple containing year and month of initial date e.g. (2002, 3)
            st_date (:class:`tuple`): Tuple containing year and month of end date e.g. (2002, 5). This date is not 
                 included in the loop
    '''
    ym_st= 12*st_date[0] + st_date[1] - 1
    ym_nd= 12*nd_date[0] + nd_date[1] - 1
    for ym in range(ym_st, ym_nd):
        y, m = divmod(ym, 12)
        yield (y, m+1)
        
def make_lon_positive(da):
    '''
        File name: make_lon_positive

        Description: Adjusts longitudes to be positive

        Author: Dougie Squire
        Date created: 02/03/2018
        Python Version: 3.5

        Args:
            lon (:class:`xarray.DataArray`): An array with longitudes continates containing negative values
        Returns:
            :class:`xarray.DataArray`: An array with all positive longitude continates
    '''
    
    da['lon'] = np.where(da['lon'] < 0, da['lon']+ 360, da['lon']) 
    da = da.sortby('lon')
    
    return da
    
def convert_to_lower_freq(ds,freq):
    '''
        File name: convert_to_lower_freq

        Description: Calculates lower frequency averages from a timeseries of higher frequency (but monotonic) means.
        Limitations: Available frequencies are freq = Y -> yearly, M -> monthly, D -> daily

        Author: Dougie Squire
        Date created: 02/03/2018
        Python Version: 3.5

        Args:
            ds (:class:`xarray.Dataset` or `xarray.DataArray`): A container of variables with daily-frequency 
                time as one dimension
            freq (:class:`str`): 'Y' -> yearly, 'M' -> monthly, 'D' -> daily
        Returns:
            :class:`xarray.Dataset` or `xarray.DataArray`: A time series of the box-averaged quantity
    '''
    
    ds['time'] = np.array(ds.time.values, dtype='datetime64['+freq+']')
    ds = ds.groupby('time').mean(dim='time')
    
    return ds
    
def calc_boxavg_latlon(da,box):
    '''
        File name: calc_boxavg_latlon

        Description: Computes the average of a given quantity over a provide lat-lon region for all time
        Limitations: Longitudinal coordinates must be positive easterly

        Author: Dougie Squire
        Date created: 14/02/2018
        Python Version: 3.5

        Args:
            da (:class:`xarray.DataArray`): A 3-dimensional array containing desired quantity to average saved 
                on grid with dimensions (time, lat, lon)
            box (:class:`tuple`): containing lat and lon extremities of box: (lat_min,lat_max,lon_min,lon_max)
        Returns:
            :class:`xarray.DataArray`: A time series of the box-averaged quantity
    '''
    
    # Adjust longitudes to be positive -----
    da = make_lon_positive(da) 

    # Extract desired region -----
    da = da.where(da['lat']>box[0],drop=True) \
            .where(da['lat']<box[1],drop=True) \
            .where(da['lon']>box[2],drop=True) \
            .where(da['lon']<box[3],drop=True)

    # Average over extracted region -----
    da = da.mean(dim=('lat','lon'))
    
    # Replicated attributes -----
    # da.attrs = da.attrs
    # da.time.encoding['units'] = da.time.encoding['units']
    # da.time.encoding['calendar'] = da.time.encoding['calendar']
    
    return da
    
def make_landsea_mask(lat_des,lon_des,detailed=False):
    '''
        File name: landsea_mask
        
        Description: Returns a grid containing a land/sea mask on the provided latitude and longitude array. 
            The mask is interpolated from NCAR landsea mask.
            (see https://www.ncl.ucar.edu/Document/Functions/Shea_util/landsea_mask.shtml)
            
        Author: Dougie Squire
        Date created: 14/02/2018
        Python Version: 3.5
        
        Args:
            lat_des (:class:`xarray.DataArray`): An array of desired latitudes (usually provided from an xarray
                Dataset or DataArray, e.g. lat_des = ds['lat'])
            lon_des (:class:`xarray.DataArray`): An array of desired longitudes (usually provided from an xarray
                Dataset or DataArray, e.g. lon_des = ds['lon'])
            detailed (:class:`boolean`): Defines whether output is detailed or not, i.e.,
                detailed=True :  ocean=0, land=1, lake=2, small island=3, ice shelf=4
                detailed=False : ocean=0, land=1, lake=0, small island=1, ice shelf=1 
                
        Returns:
            :class:`xarray.DataArray`: An array of mask values on provided lat-lon grid
    '''
    
    # Load NCAR land-sea mask -----
    ds_m = xr.open_mfdataset('../mask/ncar_landsea_mask.nc')

    # Sort and adjust longitudes to be positive -----
    lat_des = np.sort(lat_des)
    lon_des = np.sort(make_lon_positive(lon_des))
    
    # Interpolate observations onto forecast grid -----
    xi,yi = np.meshgrid(ds_m.lon,ds_m.lat)
    pntsi = np.array((np.concatenate(xi),np.concatenate(yi))).transpose()
    vali = np.concatenate(ds_m.LSMASK)
    fi = interpolate.NearestNDInterpolator(pntsi,vali)
    xd,yd = np.meshgrid(lon_des,lat_des)
    pntsd = np.array((np.concatenate(xd),np.concatenate(yd))).transpose()
    mask_array = np.reshape(fi(pntsd),(xd.shape))
    
    # Return simple or detailed mask -----
    if not detailed:
        mask_array = np.where(mask_array == 2,0,mask_array) # make lakes oceans
        mask_array = np.where(mask_array == 3,1,mask_array) # make small islands land
        mask_array = np.where(mask_array == 4,1,mask_array) # make ice shelfs oceans
        
    # Store in an xarray DataArray -----
    da = xr.DataArray(mask_array, coords=[lat_des, lon_des], dims=['lat','lon'])
    
    return da


# ===================================================================================================
# REMOVE
# ===================================================================================================
def load_ncfiles(dataset, variables):
    import glob
    folder = '/OSM/CBR/OA_DCFP/data2/observations/jra55/isobaric/007_hgt/cat/'
    file = 'jra*nc'
    files = glob.glob(folder + file)
    files = sorted(files, key=lambda x: int(x.split('.')[-3]))
    gh = xr.open_mfdataset(files, parallel=True, chunks={'initial_time0_hours':3000})['HGT_GDS0_ISBL'] \
             .rename({'g0_lon_3':'lon', 'g0_lat_2':'lat', 'initial_time0_hours':'time', 'lv_ISBL1':'level'}) \
             .sel(time=slice('2000','2010'))

    folder = '/OSM/CBR/OA_DCFP/data2/observations/jra55/isobaric/011_tmp/cat/'
    file = 'jra*nc'
    files = glob.glob(folder + file)
    files = sorted(files, key=lambda x: int(x.split('.')[-3]))[1:]
    temp = xr.open_mfdataset(files, parallel=True, chunks={'initial_time0_hours':3000})['TMP_GDS0_ISBL'] \
             .rename({'g0_lon_3':'lon', 'g0_lat_2':'lat', 'initial_time0_hours':'time', 'lv_ISBL1':'level'}) \
             .sel(time=slice('2000','2010'))

    folder = '/OSM/CBR/OA_DCFP/data2/observations/jra55/isobaric/033_ugrd/cat/'
    file = 'jra*nc'
    files = glob.glob(folder + file)
    files = sorted(files, key=lambda x: int(x.split('.')[-3]))
    u = xr.open_mfdataset(files, parallel=True, chunks={'initial_time0_hours':3000})['UGRD_GDS0_ISBL'] \
          .rename({'g0_lon_3':'lon', 'g0_lat_2':'lat', 'initial_time0_hours':'time', 'lv_ISBL1':'level'}) \
          .sel(time=slice('2000','2010'))

    folder = '/OSM/CBR/OA_DCFP/data2/observations/jra55/isobaric/034_vgrd/cat/'
    file = 'jra*nc'
    files = glob.glob(folder + file)
    files = sorted(files, key=lambda x: int(x.split('.')[-3]))
    v = xr.open_mfdataset(files, parallel=True, chunks={'initial_time0_hours':3000})['VGRD_GDS0_ISBL'] \
          .rename({'g0_lon_3':'lon', 'g0_lat_2':'lat', 'initial_time0_hours':'time', 'lv_ISBL1':'level'}) \
          .sel(time=slice('2000','2010'))

    folder = '/OSM/CBR/OA_DCFP/data2/observations/jra55/isobaric/039_vvel/cat/'
    file = 'anl*nc'
    files = glob.glob(folder + file)
    files = sorted(files, key=lambda x: int(x.split('.')[-3]))
    omega = xr.open_mfdataset(files, parallel=True, chunks={'initial_time0_hours':3000})['VVEL_GDS0_ISBL'] \
              .rename({'g0_lon_3':'lon', 'g0_lat_2':'lat', 'initial_time0_hours':'time', 'lv_ISBL1':'level'}) \
              .sel(time=slice('2000','2010'))
    
    return gh, temp, u, v, omega


# ===================================================================================================
def plot_fields(data, title, vmin, vmax, headings=None, cmin=None, cmax=None,
                ncol=2, mult_row=1, mult_col=1, mult_cshift=1, contour=False, cmap='viridis', invert=False):
    """ Plots tiles of figures """
    
    matplotlib.rc('font', family='sans-serif') 
    matplotlib.rc('font', serif='Helvetica') 
    matplotlib.rc('text', usetex='false') 
    matplotlib.rcParams.update({'font.size': 12})
    
    if len(data) == 1:
        ncol=1
        
    nrow = int(np.ceil(len(data)/ncol));

    fig = plt.figure(figsize=(11*mult_col, nrow*4*mult_row))
        
    count = 1
    for idx,dat in enumerate(data):
        if ('lat' in dat.dims) and ('lon' in dat.dims):
            trans = cartopy.crs.PlateCarree()
            ax = plt.subplot(nrow, ncol, count, projection=cartopy.crs.PlateCarree(central_longitude=180))
            ax.coastlines(color='black')
            extent = [dat.lon.min(), dat.lon.max(), 
                      dat.lat.min(), dat.lat.max()]

            if contour is True:
                if cmin is not None:
                    im = ax.contourf(dat.lon, dat.lat, dat, np.linspace(vmin,vmax,21), origin='lower', transform=trans, 
                                  vmin=vmin, vmax=vmax, cmap=cmap, extend='both')
                    ax.contour(dat.lon, dat.lat, dat, np.linspace(cmin,cmax,11), origin='lower', transform=trans,
                              colors='w', linewidths=2)
                    ax.contour(dat.lon, dat.lat, dat, np.linspace(cmin,cmax,11), origin='lower', transform=trans,
                              colors='k', linewidths=1)
                else:
                    im = ax.contourf(dat.lon, dat.lat, dat, np.linspace(vmin,vmax,20), origin='lower', transform=trans, 
                                  vmin=vmin, vmax=vmax, cmap=cmap, extend='both')
            else:
                im = ax.imshow(dat, origin='lower', extent=extent, transform=trans, vmin=vmin, vmax=vmax, cmap=cmap)

            gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True)
            gl.xlines = False
            gl.ylines = False
            gl.xlabels_top = False
            if count % ncol == 0:
                gl.ylabels_left = False
            elif (count+ncol-1) % ncol == 0: 
                gl.ylabels_right = False
            else:
                gl.ylabels_left = False
                gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator([-90, 0, 90, 180])
            gl.ylocator = mticker.FixedLocator([-90, -60, 0, 60, 90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            #ax.set_extent(extent)
            if headings is not None:
                ax.set_title(headings[idx])
        else:
            ax = plt.subplot(nrow, ncol, count)
            if 'lat' in dat.dims:
                x_plt = dat['lat']
                y_plt = dat[find_other_dims(dat,'lat')[0]]
                # if dat.get_axis_num('lat') > 0:
                #     dat = dat.transpose()
            elif 'lon' in dat.dims:
                x_plt = dat['lon']
                y_plt = dat[find_other_dims(dat,'lon')[0]]
                # if dat.get_axis_num('lon') > 0:
                #     dat = dat.transpose()
            else: 
                x_plt = dat[dat.dims[1]]
                y_plt = dat[dat.dims[0]]
                
            extent = [x_plt.min(), x_plt.max(), 
                      y_plt.min(), y_plt.max()]
            
            if contour is True:
                if cmin is not None:
                    im = ax.contourf(x_plt, y_plt, dat, levels=np.linspace(vmin,vmax,21), vmin=vmin, vmax=vmax, cmap=cmap)
                    ax.contour(x_plt, y_plt, dat, levels=np.linspace(cmin,cmax,11), colors='w', linewidths=2)
                    ax.contour(x_plt, y_plt, dat, levels=np.linspace(cmin,cmax,11), colors='k', linewidths=1)
                else:
                    im = ax.contourf(x_plt, y_plt, dat, levels=np.linspace(vmin,vmax,20), vmin=vmin, vmax=vmax, cmap=cmap)
            else:
                im = ax.imshow(dat, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
                
            if count % ncol == 0:
                ax.yaxis.tick_right()
            elif (count+ncol-1) % ncol == 0: 
                ax.set_ylabel(y_plt.dims[0])
            else:
                ax.set_yticks([])
            if idx / ncol >= nrow - 1:
                ax.set_xlabel(x_plt.dims[0])
            if headings is not None:
                ax.set_title(headings[idx], fontsize=16)
            
            if invert:
                ax.invert_yaxis()

        count += 1

    plt.tight_layout()
    fig.subplots_adjust(bottom=mult_cshift*0.16)
    cbar_ax = fig.add_axes([0.15, 0.13, 0.7, 0.020])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both');
    cbar_ax.set_xlabel(title, rotation=0, labelpad=15);
    cbar.set_ticks(np.linspace(vmin,vmax,5))
