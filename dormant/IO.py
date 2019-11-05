import numpy as np
import xarray as xr
import sys
from scipy import interpolate

#=========================================================================
class Constants(object):
	R = 287.04  # gas constant in dry air [J/K/kg]
	cp = 1004.6 # specific heat at constant pressure in dry air [J/K/kg]
	kappa = R/cp
	g = 9.80616 # nominal gravity on Earth [m/s^2]
	a = 6371000 # radius of the Earth [m]
	def __init__(self,):
		return


#=========================================================================
# General IO
#=========================================================================
def read_dataset(file_selection, data_source=None):
	ds = xr.open_mfdataset(file_selection, lock=True, autoclose=True)
	if data_source is not None:
		var_dict = get_var_dict(data_source)
		for specific_variable_name in list(ds.variables):
			if specific_variable_name in var_dict.keys():
				generic_variable_name = var_dict[specific_variable_name]
				ds = ds.rename({specific_variable_name:generic_variable_name})
	if data_source=='MOM_restart':
		ds['ps'] = xr.DataArray( ds.DELP.sum('lev'), dims=['time','lat','lon'])
	return ds


#=========================================================================
def get_var_dict(data_source):
	if data_source=='JRA55_isobaric':
		var_dict = dict()
		var_dict['initial_time0_hours'] = 'time'
		var_dict['lv_ISBL1']           = 'lev'
		var_dict['g0_lat_1']           = 'lat'
		var_dict['g0_lat_2']           = 'lat'
		var_dict['g0_lon_2']           = 'lon'
		var_dict['g0_lon_3']           = 'lon'
		var_dict['PRES_GDS0_SFC']      = 'ps'
		var_dict['GP_GDS0_SFC']        = 'orog'
		var_dict['UGRD_GDS0_ISBL']     = 'u'
		var_dict['VGRD_GDS0_ISBL']     = 'v'
		var_dict['TMP_GDS0_ISBL']      = 'temp'
		var_dict['SPFH_GDS0_ISBL']     = 'sphu'	

	elif data_source=='JRA55_hybrid':
		var_dict = dict()
		var_dict['initial_time0_hours'] = 'time'
		var_dict['lv_HYBL1']           = 'lev'
		var_dict['g4_lat_1']           = 'lat'
		var_dict['g4_lat_2']           = 'lat'
		var_dict['g4_lon_2']           = 'lon'
		var_dict['g4_lon_3']           = 'lon'
		var_dict['PRES_GDS4_SFC']      = 'ps'
		var_dict['GP_GDS4_SFC']        = 'orog'
		var_dict['UGRD_GDS4_HYBL']     = 'u'
		var_dict['VGRD_GDS4_HYBL']     = 'v'
		var_dict['TMP_GDS4_HYBL']      = 'temp'
		var_dict['SPFH_GDS4_HYBL']     = 'sphu'	

	elif data_source=='MOM_restart':
		var_dict = dict()
		var_dict['Time']  = 'time'
		var_dict['ak']    = 'aa'
		var_dict['bk']    = 'bb'
		var_dict['pfull'] = 'lev'
		var_dict['lat']   = 'lat'
		var_dict['latu']  = 'lat2'
		var_dict['lon']   = 'lon'
		var_dict['lonv']  = 'lon2'
		var_dict['Surface_pressure']     = 'ps'
		var_dict['Surface_geopotential'] = 'orog'
		var_dict['U']     = 'u'
		var_dict['V']     = 'v'
		var_dict['T']     = 'temp'
		var_dict['sphum'] = 'sphu'

	elif data_source=='MOM_daily':
		var_dict = dict()
		var_dict['Time']  = 'time'
		var_dict['ak']    = 'aa'
		var_dict['bk']    = 'bb'
		var_dict['pfull'] = 'lev'
		var_dict['lat']   = 'lat'
		var_dict['lon']   = 'lon'
		var_dict['Surface_pressure']     = 'ps'
		var_dict['Surface_geopotential'] = 'orog'
		var_dict['ucomp']     = 'u'
		var_dict['vcomp']     = 'v'
		var_dict['temp']     = 'temp'
		var_dict['sphum'] = 'sphu'		

	elif data_source=='ECMWF':	
		var_dict = dict()
		var_dict['time']  = 'time'
		var_dict['ak']    = 'aa'
		var_dict['bk']    = 'bb'
		var_dict['lev'] = 'lev'
		var_dict['lat']   = 'lat'
		var_dict['lon']   = 'lon'
		var_dict['var152']     = 'ps'
		var_dict['var129'] = 'orog'
		var_dict['var131']     = 'u'
		var_dict['var132']     = 'v'
		var_dict['var130']     = 'temp'
		var_dict['var133'] = 'sphu'

	else:
		print('specified data source "{0}" not defined')
		sys.exit()
	return var_dict


#=========================================================================
def write_binary(output_filename,field):
	field = np.reshape(field,(np.size(field),1))
	f = open(output_filename,'wb')
	field.tofile(f)
	f.close()
	return


#=========================================================================
def read_binary(input_filename, Ncol):
	f = open(input_filename,'rb')
	field = np.fromfile(f,dtype='float64')
	N=int(len(field)/Ncol)
	if Ncol>1:
		field = np.reshape(field,(Ncol,N))
	f.close()
	return field


#=========================================================================
def read_ascii(inputfilename,Ncol):
	f = open(inputfilename, 'r')
	N = sum(1 for line in f)
	f.close()

	f = open(inputfilename, 'r')
	field = np.array(np.zeros((Ncol,N), dtype=np.float64))
	j=0
	for line in f:
		line = line.strip()
		line_list = line.split()
		for i in range(0,Ncol):
			field[i,j] = float(line_list[i])
		j=j+1
	f.close()
	return field


#=========================================================================
# Interpolation
#=========================================================================
def get_lon_name(da):
	if 'lon' in da.dims:
		return 'lon'
	elif 'lon2' in da.dims:
		return 'lon2'
	else:
		print('longitude dimension not defined')
		sys.exit()


#=========================================================================
def get_lat_name(da):
	if 'lat' in da.dims:
		return 'lat'
	elif 'lat2' in da.dims:
		return 'lat2'
	else:
		print('latitude dimension not defined')
		sys.exit()


#=========================================================================
def interpolate_longitudinal_grid(ds, lon_new, var_name):
	print('\n   interpolating ',var_name)
	lon_name        = get_lon_name(ds[var_name])
	lon_axis        = ds[var_name].dims.index(lon_name)
	lon             = ds[lon_name].values
	field           = ds[var_name].values
	lon_wrap        = np.concatenate((lon-360,lon,lon+360))
	field_wrap      = np.concatenate((field,field,field),axis=lon_axis) # note wrapping around grid
	interp_function = interpolate.interp1d(lon_wrap, field_wrap, kind='linear', axis=lon_axis, bounds_error=True)
	field_new       = interp_function(lon_new)

	ds_out          = xr.Dataset()
	ds_out['time']  = ds.time.copy(deep=True)
	ds_out['lon']   = xr.DataArray( lon_new, coords=[lon_new], dims=['lon'])

	lat_name        = get_lat_name(ds[var_name])
	lat             = ds[lat_name].values
	ds_out[lat_name]= xr.DataArray( lat,     coords=[lat],     dims=[lat_name])

	if 'lev' in ds[var_name].dims:
		ds_out['lev']    = xr.DataArray( ds.lev.values, coords=[ds.lev.values], dims=['lev'])
		ds_out[var_name] = xr.DataArray( field_new, dims=['time','lev',lat_name,'lon'])
	else:
		ds_out[var_name] = xr.DataArray( field_new, dims=['time',lat_name,'lon'])
	return ds_out


#=========================================================================
def interpolate_meridional_grid(ds, lat_new, var_name):
	print('\n   interpolating ',var_name)
	lat_name        = get_lat_name(ds[var_name])
	lat_axis        = ds[var_name].dims.index(lat_name)
	lat             = ds[lat_name].values
	field           = ds[var_name].values
	lat_wrap        = np.concatenate((lat-180,lat,lat+180))
	field_flip      = np.flip(field,axis=lat_axis)
	field_wrap      = np.concatenate((field_flip,field,field_flip),axis=lat_axis) # note flipping latitdinal axis at boundaries
	interp_function = interpolate.interp1d(lat_wrap, field_wrap, kind='linear', axis=lat_axis, bounds_error=True)
	field_new       = interp_function(lat_new)

	ds_out          = xr.Dataset()
	ds_out['time']  = ds.time.copy(deep=True)
	ds_out['lat']   = xr.DataArray( lat_new, coords=[lat_new], dims=['lat'])

	lon_name        = get_lon_name(ds[var_name])
	lon             = ds[lon_name].values
	ds_out[lon_name]= xr.DataArray( lon,     coords=[lon],     dims=[lon_name])

	if 'lev' in ds[var_name].dims:
		ds_out['lev']    = xr.DataArray( ds.lev.values, coords=[ds.lev.values], dims=['lev'])
		ds_out[var_name] = xr.DataArray( field_new, dims=['time','lev','lat',lon_name])
	else:
		ds_out[var_name] = xr.DataArray( field_new, dims=['time','lat',lon_name])
	return ds_out


#=========================================================================
def convert_coefficient_from_full_to_half_levels(coef_full):
	nz = len(coef_full)+1
	coef_half = np.array(np.zeros((nz),dtype=np.float64))
	coef_half[0] = 0.0
	for i in range(0,nz-1):
		coef_half[i+1] = 2*coef_full[i] - coef_half[i]
	return coef_half


#=========================================================================
def convert_coefficient_from_half_to_full_levels(coef_half):
	return (coef_half[1:]+coef_half[:-1])/2.0


#=========================================================================
def interpolate_to_sigma_grid(ds, sigma_full_interp, var_name):
	sigma_half_interp = convert_coefficient_from_full_to_half_levels(sigma_full_interp)
	aa_interp = sigma_half_interp*0.0
	bb_interp = sigma_half_interp
	return interpolate_to_hybrid_grid(ds, aa_interp, bb_interp, var_name)


#=========================================================================
def interpolate_to_pressure_grid(ds, pfull_interp, var_name):
	phalf_interp = convert_coefficient_from_full_to_half_levels(pfull_interp)
	aa_interp = phalf_interp
	bb_interp = phalf_interp*0.0
	return interpolate_to_hybrid_grid(ds, aa_interp, bb_interp, var_name)


#=========================================================================
def interpolate_to_hybrid_grid(ds, aa_interp, bb_interp, var_name):
	# get numpy arrays from xarray object
	aa 	         = ds.aa.values
	bb	         = ds.bb.values
	ps 	         = ds.ps.values[0,:,:]
	field            = ds[var_name].values

	# do interpolation
	field_interp = interpolate_to_hybrid_grid_np(aa, bb, ps, field, aa_interp, bb_interp)

	# pass numpy arrays back into a new xarray object
	nz_interp        = len(aa_interp)
	ds_out           = xr.Dataset()
	ds_out['time']   = ds.time.copy(deep=True)
	lev_half         = np.linspace(1,nz_interp,nz_interp)
	ds_out['lev_half'] = xr.DataArray( lev_half, coords=[lev_half], dims=['lev_half'])
	ds_out['aa']     = xr.DataArray( aa, dims=['lev_half'])
	ds_out['bb']     = xr.DataArray( bb, dims=['lev_half'])	
	if (np.sum(aa_interp)>1) and (np.sum(bb_interp)>1):
		lev = np.linspace(1,nz_interp-1,nz_interp-1)
	elif np.sum(aa_interp)>1:
		lev = (aa_interp[1:]+aa_interp[:-1])/2.0
	else:
		lev = (bb_interp[1:]+bb_interp[:-1])/2.0	
	ds_out['lev']    = xr.DataArray( lev, coords=[lev], dims=['lev'])
	ds_out['lat']    = ds.lat.copy(deep=True)
	ds_out['lon']    = ds.lon.copy(deep=True)
	ds_out[var_name] = xr.DataArray( field_interp, dims=['time','lev','lat','lon'])

	return ds_out
	

#=========================================================================
def interpolate_to_hybrid_grid_np(aa, bb, ps, field, aa_interp, bb_interp):	
	nt,nz,ny,nx  = np.shape(field)
	nz_interp    = len(aa_interp)-1
	field_interp = np.array(np.zeros((nt,nz_interp,ny,nx),dtype=np.float64))
	for i in range(0,ny):
		for j in range(0,nx):
			phalf 		= aa + bb*ps[i,j]
			pfull 		= (phalf[1:]+phalf[:-1])*0.5
			log_pfull       = np.log(pfull)

			phalf_interp 	 = aa_interp + bb_interp*ps[i,j]
			pfull_interp 	 = (phalf_interp[1:]+phalf_interp[:-1])*0.5
			log_pfull_interp = np.log(pfull_interp)
			if (log_pfull_interp[-1]>log_pfull[-1]):
				# Allow extrapolation in the far field
				interp_function    = interpolate.interp1d(log_pfull, field[0,:,i,j], kind='linear', bounds_error=False, fill_value='extrapolate')
			else:
				# Do not allow extrapolation near the surface
				interp_function    = interpolate.interp1d(log_pfull, field[0,:,i,j], kind='linear', bounds_error=False, fill_value=(field[0,0,i,j],field[0,-1,i,j]))
			field_interp[0,:,i,j] = interp_function(log_pfull_interp)
	return field_interp


