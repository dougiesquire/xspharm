"""
    Contains earlier versions of functions or redundant functions that are not currently useful, but may be useful in the future
    Author: Dougie Squire
    Date created: 09/03/2018
    Python Version: 3.5
"""
        
#=============================================================================================

def convert_to_monthly_avg(ds):
    '''
        File name: convert_to_monthly_avg.py

        Description: Calculates monthly averages from a timeseries of higher frequency (but monotonic) means.
            As for mom model output, each time instance is defined at day 16, 12:00
        Limitations: 

        Author: Dougie Squire
        Date created: 02/03/2018
        Python Version: 3.5

        Args:
            ds (:class:`xarray.Dataset` or `xarray.DataArray`): A container of variables with daily-frequency 
                time as one dimension
        Returns:
            :class:`xarray.Dataset` or `xarray.DataArray`: A container of variables with monthly-frequency time
    '''

    # Loop over each year and group by month -----
    ds_y = ds.groupby('time.year')
    first_year = True
    for yr, group_y in ds_y:
        
        # Average quantities over month -----
        ds_m = group_y.groupby('time.month').mean('time',skipna=True)
        
        # Generate appropriate datetime64 timearray -----
        time_array = np.array([dt.datetime(yr,ds_m.month[i],16,12,0,0) for i in range(len(ds_m.month))], 
                              dtype='datetime64[ns]')
        ds_m['month'] = time_array
        ds_m = ds_m.rename({'month':'time'})

        # Concatenate data from each year along time dimension -----
        if first_year:
            ds_o = ds_m
            first_year = False
        else: ds_o = xr.concat([ds_o,ds_m],'time')

    ds_o.attrs = ds.attrs
    ds_o.time.encoding['units'] = ds.time.encoding['units']
    ds_o.time.encoding['calendar'] = ds.time.encoding['calendar']
    
    return ds_o
        
#=============================================================================================

def interpolate:
    x_int, y_int = np.meshgrid(ds_observat_resampled.lon,ds_observat_resampled.lat)

    observat_interp = np.zeros([len(ds_forecast_resampled.time),
                                len(ds_forecast_resampled.lat),
                                len(ds_forecast_resampled.lon)])

    for t in [1]: #range(len(ds_observat_resampled.time)):
        # Set up interpolation -----
        val_int = squeeze(ds_observat_resampled.isel(time=[t]))
        print(val_int)
        f = sp.interpolate.interp2d(x_int, y_int, val_int, kind = 'linear')

        x_des, y_des = np.meshgrid(ds_forecast_resampled.lon,ds_forecast_resampled.lat)
        observat_interp[t,:,:] = f(x_des, y_des)
        
#=============================================================================================

def histogram_gufunc(x,bins):
    ''' Computes histogram from rank data '''

    print(x)
    print(bins)
    print(np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], -1, x))
    return np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], -1, x)

def histogram_data(ranked,bin_edges,dim='init_date'): 
    ''' Computes histogram of data along specified dimension '''
    
    return xr.apply_ufunc(histogram_gufunc, ranked, bin_edges,
                          input_core_dims=[[dim],[]],
                          dask='allowed',
                          output_dtypes=[int])

with timer():
    # Calculate the histogram for each lead time -----
    bin_edges = linspace(0,len(ensembles)+1,len(ensembles)+2) + 0.5
    bins = linspace(1,len(ensembles)+1,len(ensembles)+1)
    
    da_histogram = histogram_data(da_ranked,bin_edges)

da_histogram

#=============================================================================================

def histogram_gufunc(x, bins):
    ''' Returns histogram of ranked data along specified dimension '''
    # print(x)
    # print(bins)
    test = np.apply_along_axis(lambda x: np.histogram(x, bins=bins)[0], -1, x)
    # print(test)
    return np.apply_along_axis(lambda x: np.histogram(x, bins=bins)[0], -1, x)

def compute_histogram(da, bin_edges, dim='init_date'):
    """ Feeds ranked data to ufunc that computes histogram """
    # Pad input array to be same size as histogram so apply_ufunc can be used -----
    da_nan = da.copy()
    print(da_nan)
    print(len(bin_edges)-len(da[dim]))
    return xr.apply_ufunc(histogram_gufunc, da, bin_edges,
                          input_core_dims=[[dim],[]],
                          dask='allowed',
                          output_dtypes=[int]).rename('histogram')

# Compute the histogram(s) -----
bin_edges = linspace(0,len(ensembles)+1,len(ensembles)+2) + 0.5
bins = linspace(1,len(ensembles)+1,len(ensembles)+1)
histogram_data = compute_histogram(da_ranked, bin_edges, dim='init_date')

#=============================================================================================

def compute_histogram(da, bin_edges, dim='init_date'):
    """ Returns histogram along specified dimension """
    ax = da.get_axis_num(dim) 
    return np.apply_along_axis(lambda x: np.histogram(x, bins=bin_edges)[0], ax, da)

# Compute the histogram(s) -----
bin_edges = linspace(0,len(ensembles)+1,len(ensembles)+2) + 0.5
bins = linspace(1,len(ensembles)+1,len(ensembles)+1)
histogram_data = compute_histogram(da_ranked, bin_edges, dim='init_date')

#=============================================================================================

def calc_dicontingency(fcst,obsv,indep_dim):
    """ Construct dichotomous contingency table from logical event data """
    
    hits = ((fcst == True) & (obsv == True)).sum(dim=indep_dim)
    false_alarms = ((fcst == True) & (obsv == False)).sum(dim=indep_dim)
    misses = ((fcst == False) & (obsv == True)).sum(dim=indep_dim)
    correct_negs = ((fcst == False) & (obsv == False)).sum(dim=indep_dim)
    
    total_yes_fcst = (fcst == True).sum(dim=indep_dim)
    total_no_fcst = (fcst == False).sum(dim=indep_dim)
    total_yes_obsv = (obsv == True).sum(dim=indep_dim)
    total_no_obsv = (obsv == False).sum(dim=indep_dim)
    total = ((fcst == True) | (fcst == False)).sum(dim=indep_dim)
    
    # Package in Dataset -----
    contingency = hits.to_dataset('hits')
    contingency.hits.attrs['name'] = 'number of hits'
    contingency['false_alarms'] = false_alarms
    contingency.false_alarms.attrs['name'] = 'number of false alarms'
    contingency['misses'] = misses
    contingency.misses.attrs['name'] = 'number of misses'
    contingency['correct_negs'] = correct_negs
    contingency.correct_negs.attrs['name'] = 'number of correct negatives'
    
    contingency['total_yes_fcst'] = total_yes_fcst
    contingency.total_yes_fcst.attrs['name'] = 'total number of forecast yes'
    contingency['total_no_fcst'] = total_no_fcst
    contingency.total_no_fcst.attrs['name'] = 'total number of forecast no'
    contingency['total_yes_obsv'] = total_yes_obsv
    contingency.total_yes_obsv.attrs['name'] = 'total number of observed yes'
    contingency['total_no_obsv'] = total_no_obsv
    contingency.total_no_obsv.attrs['name'] = 'total number of observed no'
    contingency['total'] = total
    contingency.total.attrs['name'] = 'overall total'
    
    return contingency
    
def compute_dicontingency_table(fcst_logical,obsv_logical,ensemble_dim,indep_dim):
    """ Returns the dichotomous contingency table from logical event data """
    return fcst_logical.groupby(ensemble_dim) \
                       .apply(calc_dicontingency,obsv=obsv_logical,indep_dim=indep_dim) \
                       .sum(ensemble_dim)
        

#=============================================================================================
def downsample_complete(data, resample_freq, how, input_freq=None):
    """ 
    Resamples provided xarray object along time dimension at specified frequency using specified method. 
    Only complete frequency intervals are retained.
    Frequency options: 'D' (daily); 'W' (weekly); 'M' (monthly); 'A' (annual).
    
    DO NOT try to specify numbers of increments (e.g. '6M') for the resampling frequency, as this function
    uses xarray.resample() which has bugs for multiple months and years
    """
        
    data_sampled = data.resample(time=resample_freq)
    print(list(data_sampled))
    
    # Ensure only recognised frequencies are provided -----
    if not (('D' in resample_freq) | ('W' in resample_freq) | ('M' in resample_freq) | ('A' in resample_freq)):
        raise ValueError(f'{resample_freq} is not a recognised requency')

    # Try to determine the input frequency otherwise ask the user to specify this -----
    if input_freq == None:
        input_freq = utils.infer_freq(data.time.values)
    if input_freq == None:
        raise ValueError('Cannot infer input frequency. Please specify this explicitly using function input, input_freq')
    if not (('D' in input_freq) | ('W' in input_freq) | ('M' in input_freq) | ('A' in input_freq)):
        raise ValueError(f'{input_freq} is not a recognised requency')
        
    # Find which resample bins are complete -----
    keep_bin = np.zeros(len(data_sampled))
    loop_count = 0
    for sample_bin, data_sample in data_sampled:
        sample = data_sample.time.values[0]
        print(data_sample,'\n')
        
        # Find resample period -----
        if 'M' in resample_freq:
            # Check if multiple months are specified
            if len(resample_freq) > 1:
                raise ValueError('Multi-month functionality not available due to limitations with xarray.resample()')
            else: num_months = 1
            
            resample_range = [utils.trunc_time(sample, freq='M')]
            resample_range.append(month_delta(resample_range[0],num_months))
        elif 'A' in resample_freq:
            # Check if multiple years are specified
            if len(resample_freq) > 1:
                raise ValueError('Multi-year functionality not available due to limitations with xarray.resample()')
            else: num_years = 1
            
            resample_range = [utils.trunc_time(sample, freq = 'Y')]
            resample_range.append(year_delta(resample_range[0], num_years))
        else:
            resample_range = (pd.date_range(sample, periods=2, freq=resample_freq)).values
        
        resample_period = resample_range[1] - resample_range[0]
        
        # Find input period -----
        if 'M' in input_freq:
            # Check if multiple months are specified
            if len(input_freq) > 1:
                num_months = int(input_freq.replace("M", ""))
            else: num_months = 1
            
            input_range = [utils.trunc_time(sample, freq='M')]
            input_range.append(month_delta(input_range[0],num_months))
        elif 'A' in input_freq:
            # Check if multiple years are specified
            if len(input_freq) > 1:
                num_years = int(input_freq.replace("Y", ""))
            else: num_years = 1
            
            input_range = [utils.trunc_time(sample, freq = 'Y')]
            input_range.append(month_delta(input_range[0],12 * num_years))
        else:
            input_range = (pd.date_range(sample, periods=2, freq=input_freq)).values
        
        input_period = input_range[1] - input_range[0]
        
        # Compute required number of increments in a complete resample period -----
        min_num_incr = np.floor(resample_period / input_period)
        max_num_incr = np.ceil(resample_period / input_period)
        n_sample = len(data_sample.time)
        
        # Only keep resample bins that are complete -----
        first_step = data_sample.time.values[0] - resample_range[0]
        if ((n_sample == min_num_incr) | (n_sample == max_num_incr)) & (first_step < input_period):
            keep_bin[loop_count] = 1
        
        loop_count += 1
            
    print(keep_bin)
    
    if how == 'mean':
        data_resampled = data_sampled.mean()
    elif how == 'sum':
        data_resampled = data_sampled.sum()
    else:
        raise ValueError(f'Unrecognised "how" method: {how}')
    
    # Truncate to integer increment of resample frequency -----
    if 'A' in resample_freq:
        trunc_freq = 'Y'
    else:
        trunc_freq =''.join([i for i in resample_freq if not i.isdigit()])
    
    data_resampled['time'] = utils.trunc_time(data_resampled['time'], trunc_freq)
    
    return data_resampled

#=============================================================================================
def month_delta(date_in, delta, trunc_to_start=False):
    """ Increments provided datetime64 array by delta months """
    
    date_mod = pd.Timestamp(date_in)
    
    m, y = (date_mod.month + delta) % 12, date_mod.year + ((date_mod.month) + delta - 1) // 12
    if not m: m = 12
    d = min(date_mod.day, [31,
        29 if y % 4 == 0 and not y % 400 == 0 else 28,31,30,31,30,31,31,30,31,30,31][m - 1])
    
    if trunc_to_start:
        date_out = utils.trunc_time(np.datetime64(date_mod.replace(day=d,month=m, year=y)),'M')
    else:
        date_out = np.datetime64(date_mod.replace(day=d,month=m, year=y))
    return date_out


#=============================================================================================
def year_delta(date_in, delta, trunc_to_start=False):
    """ Increments provided datetime64 array by delta years """
    
    date_mod = month_delta(date_in, 12 * delta)
    
    if trunc_to_start:
        date_out = utils.trunc_time(date_mod,'Y')
    else: date_out = date_mod
        
    return date_out


#=============================================================================================
def leadtime_to_datetime(data, lead_time_name='lead_time', init_date_name='init_date'):
    """ Converts time information from lead time/initial date dimension pair to single datetime dimension """
    
    init_date = data[init_date_name].values[0]
    lead_times = list(map(int,data[lead_time_name].values))
    freq = data[lead_time_name].attrs['units']

    # Deal with special cases of monthly and yearly frequencies -----
    if 'M' in freq:
        # Check if multiple months are specified
        if len(freq) > 1:
            num_months = int(freq.replace("M", ""))
        else: num_months = 1
            
        datetimes = np.array([month_delta(init_date, num_months * ix) for ix in lead_times])
    elif 'A' in freq:
        # Check if multiple months are specified
        if len(freq) > 1:
            num_years = int(freq.replace("A", ""))
        else: num_years = 1
            
        datetimes = np.array([year_delta(init_date, num_years * ix) for ix in lead_times])
    else:
        datetimes = (pd.date_range(init_date, periods=len(lead_times), freq=freq)).values
    
    data = data.drop(init_date_name)
    data = data.rename({lead_time_name : 'time'})
    data['time'] = datetimes
    
    return utils.prune(data)


#=============================================================================================
def downsample_complete(data, resample_freq, how, input_freq=None):

    dates = data.time.values

    # Try to infer input frequency -----
    if input_freq == None:
        input_freq = utils.infer_freq(dates)
    if input_freq == None:
        raise ValueError('Unable to infer input frequency. Please specify this explicity.')

    # Split frequencies into numbers and strings -----
    incr_string = ''.join([i for i in resample_freq if i.isdigit()])
    resample_incr = [int(incr_string) if incr_string else 1][0]
    resample_type = ''.join([i for i in resample_freq if not i.isdigit()])

    incr_string = ''.join([i for i in input_freq if  i.isdigit()])
    input_incr = [int(incr_string) if incr_string else 1][0]
    input_type = ''.join([i for i in input_freq if not i.isdigit()])

    # Construct dummy date array to determine complete number of increments in each resample bin -----
    if 'M' in resample_type: # Deal with special case of months
        start = month_delta(dates[0],-resample_incr)
        end = month_delta(dates[-1],resample_incr)

        # Ensure dummy_dates align with dates in overlap region -----
        left_chunk = (pd.date_range(start, dates[0], freq = input_freq)).values
        left_chunk_aligned = left_chunk + (dates[0] - left_chunk[-1])
        right_chunk_aligned = (pd.date_range(dates[-1], end, freq = input_freq)).values
        dummy_dates = np.concatenate([left_chunk_aligned, dates[1:-1], right_chunk_aligned])

    elif ('A' in resample_type) | ('Y' in resample_type): # Deal with special case of years
        start = year_delta(dates[0],-resample_incr)
        end = year_delta(dates[-1],resample_incr)

        # Ensure dummy_dates align with dates in overlap region -----
        left_chunk = (pd.date_range(start, dates[0], freq = input_freq)).values
        left_chunk_aligned = left_chunk + (dates[0] - left_chunk[-1])
        right_chunk_aligned = (pd.date_range(dates[-1], end, freq = input_freq)).values
        dummy_dates = np.concatenate([left_chunk_aligned, dates[1:-1], right_chunk_aligned])

    else:
        start = dates[0] - pd.Timedelta(resample_incr, unit = resample_type)
        end = dates[-1] + pd.Timedelta(resample_incr, unit = resample_type)

        # Ensure dummy_dates align with dates in overlap region -----
        left_chunk = (pd.date_range(start, dates[0], freq = input_freq)).values
        left_chunk_aligned = left_chunk + (dates[0] - left_chunk[-1])
        right_chunk_aligned = (pd.date_range(dates[-1], end, freq = input_freq)).values
        dummy_dates = np.concatenate([left_chunk_aligned, dates[1:-1], right_chunk_aligned])
        print(dates)
        print(dummy_dates)

    # Package dummy date array as xarray object and resample -----
    dummy = xr.DataArray(np.zeros(dummy_dates.shape), coords=[dummy_dates], dims='time')
    dummy_sampled = dummy.resample(time=resample_freq)
    data_sampled = data.resample(time=resample_freq)

    # Find and compare number of increments in each dummy bin and data bin -----
    dummy_incr = [len(dummy_bin.time) for name, dummy_bin in dummy_sampled][1:-1]
    data_incr = [len(data_bin.time) for name, data_bin in data_sampled]
    data_bins = [name for name, data_bin in data_sampled]
    keep = [dum == dat for (dum, dat) in zip(dummy_incr, data_incr)]
    print(dummy_incr)
    print(data_incr)

    # Perform resampling according to specified method -----
    if how == 'mean':
        data_resampled = data_sampled.mean(dim='time',keep_attrs=True)
    elif how == 'sum':
        data_resampled = data_sampled.sum(dim='time',keep_attrs=True)
    else:
        raise ValueError(f'Unrecognised "how" method: {how}. Please feel free to add methods.')

    # Strangely, xarray.resample().how() adds an additional time step to the beginning or end of 
    # the data (depending on whether the resample frequency is a start or end frequency) when the 
    # time interval of the data being resampled is wholly divisible by the resampling frequency.
    # Data at this time step are all nans. Let's make sure we only keep output time steps that 
    # exist in the xarray.core.resample.DataArrayResample object -----
    data_resampled = data_resampled.sel(time=slice(str(data_bins[0]), str(data_bins[-1])))

    # Only keep resample bins that are complete -----
    data_resampled = data_resampled.sel(time=data_resampled.time[keep])

    print(data_resampled)
    print(keep)

#=============================================================================================
def regrid_weighted_avg(da, lat, lon):
    """ Performs weighted average regridding """

    lon_edges = utils.get_bin_edges(lon)
    print(lon_edges)
    lat_edges = utils.get_bin_edges(lat)
    print(list(da.groupby_bins('lon',lon_edges)))
    lonreg = da.groupby_bins('lon',lon_edges).mean(dim='lon')

    regridded = lonreg.groupby_bins('lat',lat).mean(dim='lat')
    

# ===================================================================================================
def map_coordinates(source, target, keep_attrs=True, **kwargs):
    """
    Uses ``scipy.ndimage.interpolation.map_coordinates`` to map the source xarray data to the 
    given target coordinates using spline interpolation.
    Note that target represents a mapping from dimension names to coordinate values to sample at, 
    and **kwargs are passed to ``scipy.ndimage.interpolation.map_coordinates``.

    """

    # Set up the interpolators to map coordinates onto array indices
    interpolators = {}
    for dim_name in target.keys():
        dim = source.coords[dim_name]
        if not np.issubdtype(dim.dtype, np.number):
            raise ValueError('Only numerical dimensions '
                             'can be interpolated on.')
        try:
            interpolators[dim_name] = interp1d(dim, list(range(len(dim))))
        except ValueError:  # Raised if there is only one entry
            # 0 is the only index that exists
            interpolators[dim_name] = lambda x: 0

    # Set up the array indices on which to interpolate, and the final coordinates 
    indices = collections.OrderedDict()
    coords = collections.OrderedDict()
    for d in source.dims:
        if d not in target.keys():
            # Choose all entries from non-interpolated dimensions
            indices[d] = list(range(len(source.coords[d])))
            coords[d] = source.coords[d]
        else:
            # Choose selected entries from interpolated dimensions
            indices[d] = [interpolators[d](i) for i in target[d]]
            coords[d] = target[d]

    # Generate array of all coordinates
    # Shape should be n(dims) * n(product of all interpolators)
    coordinates = np.array(list(zip(
        *itertools.product(*[i for i in indices.values()]))
    ))

    interp_array = ndimage.map_coordinates(source.values, coordinates,
                                           **kwargs)

    # Reshape the resulting array according to the target coordinates
    result_shape = [len(i) for i in indices.values()]
    attrs = source.attrs if keep_attrs else None  # Copy attrs if asked for
    return DataArray(interp_array.reshape(result_shape),
                     coords=coords, attrs=attrs)


# ===================================================================================================
def cdoize(ds, var_name, lat_name, lon_name, time_name, save_dir):
    """ Saves var-lat-lon-time Dataset as temporary .nc file in cdo-anticipated format """
    
    temp = xr.Dataset(data_vars={'sst' : (['time', 'lat', 'lon'], ds[var_name], {'standard_name' : var_name,
                                                                  'long_name' : var_name}),
                                           'time' : ('time', ds[time_name], {'standard_name' : time_name,
                                                                            'long_name' : time_name}),
                                           'lon' : ('lon', ds[lon_name], {'standard_name' : 'longitude',
                                                                         'long_name' : 'longitude',
                                                                         'units' : 'degrees_east',
                                                                         'axis' : 'X'}),
                                           'lat' : ('lat', ds[lat_name], {'standard_name' : 'latitude',
                                                                         'long_name' : 'latitude',
                                                                         'units' : 'degrees_north',
                                                                         'axis' : 'Y'})})
    
    temp.to_netcdf(save_dir) #, encoding = {'time':{'dtype':'float','calendar':'standard',
                             #            'units':'days since 1700-01-01 00:00:00'}})


# ===================================================================================================
def gradient_regular(da, dim, method='2CD'):
    """
    Returns the gradient computed using the specified central differences method in the 
    interior points and first order accurate one-sided (forward or backwards) differences 
    at the boundaries

    Data is returned on the same grid only for regular grids
    """
    
    if method == '2CD':
        centre_chunk = range(len(da[dim])-2)
        
        num_c = (da.shift(**{dim:-2}) - da).isel(**{dim : centre_chunk})
        num_c[dim] = ((da[dim].shift(**{dim:-2}) + da[dim])/2).isel(**{dim : centre_chunk})
        den_c = (da[dim].shift(**{dim:-2}) - da[dim]).isel(**{dim : centre_chunk})
        den_c[dim] = num_c[dim]

        num_l = (da.shift(**{dim:-1}) - da).isel(**{dim : 0})
        den_l = (da[dim].shift(**{dim:-1}) - da[dim]).isel(**{dim : 0})

        num_r = (-da.shift(**{dim:1}) + da).isel(**{dim : -1})
        den_r = (-da[dim].shift(**{dim:1}) + da[dim]).isel(**{dim : -1})
        
    elif method == '4CD':
        centre_chunk = range(len(da[dim])-4)
        
        num_c = (da - da.shift(**{dim:-4}) + 8 * da.shift(**{dim:-3}) - 
                 8 * da.shift(**{dim:-1})).isel(**{dim : centre_chunk})
        num_c[dim] = ((da[dim] + da[dim].shift(**{dim:-4}))/2).isel(**{dim : centre_chunk})
        den_c = (da[dim] - da[dim].shift(**{dim:-4}) + 8 * da[dim].shift(**{dim:-3}) - 
                 8 * da[dim].shift(**{dim:-1})).isel(**{dim : centre_chunk})
        den_c[dim] = num_c[dim]
        
        num_l = (da.shift(**{dim:-1}) - da).isel(**{dim : [0,1]})
        den_l = (da[dim].shift(**{dim:-1}) - da[dim]).isel(**{dim : [0,1]})

        num_r = (-da.shift(**{dim:1}) + da).isel(**{dim : [-2,-1]})
        den_r = (-da[dim].shift(**{dim:1}) + da[dim]).isel(**{dim : [-2,-1]})
 
    else:
        raise TypeError('Unrecognised gradient method')
    
    # Combine -----
    grad = xr.concat([num_l, num_c, num_r], dim=dim) / \
           xr.concat([den_l, den_c, den_r], dim=dim)
    
    # Check that points are ending up where they started -----
    tol = 1e-10
    if not np.all(abs(da[dim].values - grad[dim].values) < tol):
        raise ValueError('Output grid differs from input grid')
    
    grad[dim] = da[dim]
    
    return grad
           