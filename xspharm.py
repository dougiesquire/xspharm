# Adadpted from spencerclark's code base: https://gist.github.com/spencerkclark/6a8e05a492111e52d8d8fb407d332611
import spharm
import numpy as np
import xarray as xr
import dask.array as darray
from spharm import Spharmt

_HARMONIC_DIM = 'harmonic'
_TOTAL_WAVENUMER_DIM = 'n'
_LONGITUDINAL_WAVENUMER_DIM = 'm'
_NON_HORIZONTAL_DIM = 'non_horizontal'
RADIUS = 6370997.

# ===================================================================================================
# Utilities
# ===================================================================================================

def wraps_dask_array(da):
    """
        Check if an xarray object wraps a dask array
    """
    return isinstance(da.data, darray.core.Array)

def get_other_dims(da, dim_exclude):
    """ 
        Returns all dimensions in provided xarray object excluding dim_exclude 
    """
    
    dims = da.dims
    
    if dims_exclude == None:
        return dims
    else:
        if isinstance(dims, str):
            dims = [dims]
        if isinstance(dims_exclude, str):
            dims_exclude = [dims_exclude]

        return set(dims) - set(dims_exclude)

def flip_lat(da, lat_name):
    """
        Flip latitude dimension
    """
    return da.isel(**{lat_name: slice(None, None, -1)})


def create_spharmt(n_lon, n_lat, gridtype):
    """
        Initialise Spharmt object
    """
    return Spharmt(n_lon, n_lat, rsphere=RADIUS, gridtype=gridtype)


def stack_non_horizontal_dims(da, non_horizontal_dims):
    """
        If present, stack all non-horizontal dims onto one dimension
    """
    dims_to_stack = get_other_dims(da, non_horizontal_dims)
    if dims_to_stack:
        da = da.stack(**{_NON_HORIZONTAL_DIM: dims_to_stack})
    return da


def N_harmonics(n_trunc):
    """
        Return the number of harmonics
    """
    return (n_trunc + 1) * (n_trunc + 2) // 2


def order_dims_first(da, first_dims):
    """
        Order dims such that first_dims come first
    """
    order = first_dims + get_other_dims(da, first_dims)
    return da.transpose(*order)


def make_single_chunk(da, dims):
    """
        If underlying data are chunked, rechunk specified dims to single chunk
    """
    chunks = {dim : -1 for dim in dims}

    if wraps_dask_array(da):
        da = da.chunk(chunks)
    return da


def add_attrs(da, **kwargs):
    """
        Add attributes to xarray object
    """
    for key, value in kwargs.items():
        if key in da.attrs:
            da.attrs[key] = da.attrs[key] + value
        else:
            da.attrs[key] = value
    return da


def get_spharm_grid(n_lat, gridtype): 
    """
        Generate spharm latitude-longitude grid.
    """
    if gridtype == 'gaussian':
        lat = spharm.gaussian_lats_wts(n_lat)[0]
        lon = np.linspace(0,360,2*len(lat)+1)[0:-1]
    elif gridtype == 'regular':
        if n_lat % 2 == 0:
            lat = np.linspace(90-90/n_lat, -90+90/n_lat, n_lat)
        else:
            lat = np.linspace(90, -90, n_lat)
        lon = np.linspace(0,360,2*len(lat)+1)[0:-1]
    else:
        raise ValueError('Unrecognised gridtype')
    return lat, lon


def repack_mn(da, n_trunc):
    """
        Pack m-n wavenumber pairs into single dimension ordered as expected by spharm.spectogrd
    """
    to_concat = []
    prev = 0
    for m in range(n_trunc+1):
        da_h = da.sel({_LONGITUDINAL_WAVENUMER_DIM : m}, drop=True).sel({_TOTAL_WAVENUMER_DIM : slice(m,n_trunc+1)}) \
                 .rename({_TOTAL_WAVENUMER_DIM:_HARMONIC_DIM})
        da_h[_HARMONIC_DIM] = range(prev, prev+n_trunc-m+1)
        to_concat.append(da_h)
        prev += n_trunc-m+1

    return xr.concat(to_concat, dim=_HARMONIC_DIM).chunk({_HARMONIC_DIM:-1})


def unpack_mn(da, n_trunc):
    """
        Unpack output from grdtospec into m-n wavenumber pairs
    """
    to_concat = []
    prev = 0
    for n in range(n_trunc+1):
        da_n = da.isel({_HARMONIC_DIM:slice(prev,prev+n_trunc-n+1)}) \
                   .rename({_HARMONIC_DIM:_TOTAL_WAVENUMER_DIM})
        da_n[_TOTAL_WAVENUMER_DIM] = range(n, n_trunc+1)
        to_concat.append(da_n)
        prev += n_trunc-n+1

    unpacked = xr.concat(to_concat, dim=_LONGITUDINAL_WAVENUMER_DIM)
    unpacked[_LONGITUDINAL_WAVENUMER_DIM] = range(0, n_trunc+1)
    return unpacked


def sum_along_m(coeffs):
    """
        Returns the sum along the longitudinal wavenumber dimension of the provided coefficients \
            computed using spharm, accounting for the fact that spharm returns one side of the \
            decomposition
            
        Parameters
        ----------
        coeffs : xarray DataArray
            Array containing spherical harmonic coefficients
        
        Returns
        -------
        xr.DataArray
            Array containing the sum of the coefficients along the longitudinal wavenumber
    """
    if _HARMONIC_DIM in coeffs.dims:
        coeffs = unpack_mn(coeffs, n_trunc)
    return xr.concat([coeffs.sel(m=0), 
                      2*coeffs.sel(m=slice(1,len(coeffs.m)+1))], dim='m').sum('m')


def mean_along_m(coeffs):
    """
        Returns the mean along the longitudinal wavenumber dimension of the provided coefficients \
            computed using spharm, accounting for the fact that spharm returns one side of the \
            decomposition
            
        Parameters
        ----------
        coeffs : xarray DataArray
            Array containing spherical harmonic coefficients
        
        Returns
        -------
        xr.DataArray
            Array containing the mean of the coefficients along the longitudinal wavenumber
    """
    if _HARMONIC_DIM in coeffs.dims:
        coeffs = unpack_mn(coeffs, n_trunc)
    return xr.concat([coeffs.sel(m=0), 
                      2*coeffs.sel(m=slice(1,len(coeffs.m)+1))], dim='m').mean('m')


def get_power(coeffs):
    """
        Returns the power, S(n), given the spherical harmonic coefficients computed using xspharm
        
        Parameters
        ----------
        coeffs : xarray DataArray
            Array containing spherical harmonic coefficients
        
        Returns
        -------
        xr.DataArray
            Array containing the power, S(n)
    """
    if _HARMONIC_DIM in coeffs.dims:
        coeffs = unpack_mn(coeffs, n_trunc)
    return integrate_along_m(abs(coeffs) ** 2)


def prep_for_spharm(da, lat_dim='lat', lon_dim='lon'):
    """
        Prepare DataArray for use with spharm (e.g. grdtospec)
        
        Parameters
        ----------
        da : xarray DataArray
            Input DataArray
        lat_dim : str
            Name of latitude dimension
        lon_dim : str
            Name of longitude dimension
        
        Returns
        -------
        xr.DataArray, boolean
            Array containing data that has been prepared for use with spharm, and a \
                boolean indicating whether the data was latitudinally flipped
    """

    def _orient_latitude_north_south(da, lat_dim):
        """
            Orients data such that northern latitudes come first
            Returns the transformed array as well as flag noting if the data were
            flipped.
        """
        if all(da[lat_dim].diff(lat_dim) > 0.):
            return flip_lat(da, lat_dim), True
        else:
            return da, False

    da = stack_non_horizontal_dims(da, (lat_dim, lon_dim))
    da = order_dims_first(da, (lat_dim, lon_dim))
    da = make_single_chunk(da, (lat_dim, lon_dim))
    return _orient_latitude_north_south(da, lat_dim)


def prep_for_inv_spharm(da):
    """
        Prepare DataArray for use with inverse spharm (e.g. spectogrd)
        
        Parameters
        ----------
        da : xarray DataArray
            Input DataArray
        
        Returns
        -------
        xr.DataArray, boolean
            Array containing data that has been prepared for use with spharm, and a \
                boolean indicating whether the data was latitudinally flipped
    """
    
    da = stack_non_horizontal_dims(da, (_HARMONIC_DIM, ))
    da = order_dims_first(da, (_HARMONIC_DIM, ))
    return make_single_chunk(da, (_HARMONIC_DIM, ))


# ===================================================================================================
# grdtospec
# ===================================================================================================

def grdtospec(da, gridtype, lat_dim='lat', lon_dim='lon', n_trunc=None, unpack_wavenumber_pairs=False, prepped=False):
    """
        Returns complex spherical harmonic coefficients resulting from the spherical harmonic analysis \
                of da using spharm package
        Note, the spharm coefficents are scaled such that they mush be divided by the sqrt(2) in order \
            to sum to the variance
        
        Parameters
        ----------
        da : xarray DataArray
            Array of real space data to use to compute spherical harmonics. Must contain at least latitude and \
                longitude dimensions, with either regular or Gaussian gridding. 
        gridtype : "gaussian" or "regular"
            Grid type of da
        n_trunc : int, optional
            Spectral truncation limit
        unpack_wavenumber_pairs : boolean
            Set to True to unpack coefficients onto total and longitudinal wavenumbers. Otherwise returns \
                coefficients along a single dimension, ordered as output by spharm.grdtospec
        prepped : boolean
            Set to True if data is formatted (stacked, ordered and chunked) such that it can be handed directly \
                to spharm. Default is False
            
        Returns
        -------
        xarray DataArray
            Array containing the complex spectral harmonic coefficients of da
    """

    def _grdtospec(st, da, n_trunc):
        """
            Wrap Spharmt.grdtospec to be dask compatible
        """
        if isinstance(da, darray.core.Array):
            if da.ndim == 3:
                chunks = ((N_harmonics(n_trunc), ), da.chunks[-1])
            else:
                chunks = ((N_harmonics(n_trunc), ))
            return darray.map_blocks(st.grdtospec, da, n_trunc,
                                     chunks=chunks, dtype=np.complex, 
                                     drop_axis=(0, 1), new_axis=(0, ))
        else:
            return st.grdtospec(da)
    
    if not prepped:
        da, flipped = prep_for_spharm(da)

    if n_trunc is None:
        n_trunc = int(da.sizes[lat_dim]/2) - 1
    
    st = create_spharmt(da.sizes[lon_dim], da.sizes[lat_dim], gridtype=gridtype)

    if _NON_HORIZONTAL_DIM in da.dims:
        output_core_dims = [[_HARMONIC_DIM, _NON_HORIZONTAL_DIM]]
        output_sizes = {_HARMONIC_DIM: N_harmonics(n_trunc),
                        _NON_HORIZONTAL_DIM: da.sizes[_NON_HORIZONTAL_DIM]}
        coeffs = xr.apply_ufunc(_grdtospec, st, da, n_trunc,
                                input_core_dims=[[], da.dims, []],
                                output_core_dims=output_core_dims,
                                output_sizes=output_sizes,
                                exclude_dims=set((lat_dim, lon_dim)), dask='allowed') \
                   .unstack(_NON_HORIZONTAL_DIM)
    else:
        output_core_dims = [[_HARMONIC_DIM]]
        output_sizes = {_HARMONIC_DIM: N_harmonics(n_trunc)}

        coeffs = xr.apply_ufunc(_grdtospec, st, da, n_trunc,
                                input_core_dims=[[], da.dims, []],
                                output_core_dims=output_core_dims,
                                output_sizes=output_sizes,
                                exclude_dims=set((lat_dim, lon_dim)), dask='allowed')
    
    coeffs[_HARMONIC_DIM] = range(coeffs.sizes[_HARMONIC_DIM]) 
    coeffs = add_attrs(coeffs, **{'xspharm_history':'grdtospec(..'
                                  ' gridtype=' + gridtype + 
                                  ', n_trunc=' + str(n_trunc) +
                                  ', unpack_wavenumber_pairs=' + str(unpack_wavenumber_pairs) +
                                  ', prepped=' + str(prepped) +')'})
        
    if unpack_wavenumber_pairs:
        return unpack_mn(coeffs, n_trunc)
    else:
        return coeffs

    
# ===================================================================================================
# spectogrd
# ===================================================================================================

def spectogrd(da, gridtype='gaussian', n_lat=None,  prepped=False, lat_name='lat', lon_name='lon'):
    """
        Returns the real-space fields resulting from the spherical harmonic synthesis of da using spharm package
        
        Parameters
        ----------
        da : xarray DataArray
            Array of spherical haramonic coefficients. Coefficients must either be ordered along the dimension, \
                _HARMONIC_DIM, in the order expected by spharm.spectogrd, or must be stacked according to their \
                total and longitudinal wavenumbers along dimensions _TOTAL_WAVENUMER_DIM and \
                _LONGITUDINAL_WAVENUMER_DIM, respectively
        gridtype : "gaussian" or "regular"
            Desired gridtype type of output 
        n_lat : int, optional
            Desired number of latitudes in output
        prepped : boolean, optional
            Set to True if data is formatted (stacked, ordered and chunked) such that it can be handed directly \
                to spharm. Default is False
            
        Returns
        -------
        xarray DataArray
            Array containing the complex spherical harmonic synthesis of da
    """

    def N_trunc(n_harmonics):
        return int(np.sqrt(2*n_harmonics)-1)
    
    def _spectogrd(st, da):
        """
            Transform a variable from spectral to grid space
        """
        if isinstance(da, darray.core.Array):
            if da.ndim == 2:
                chunks = ((st.nlat, ), (st.nlon, ), da.chunks[-1])
            else:
                chunks = ((st.nlat, ), (st.nlon, ))
            return darray.map_blocks(st.spectogrd, da, chunks=chunks,
                                     drop_axis=(0, ), new_axis=(0, 1))
        else:
            return st.spectogrd(da)
    
    if not prepped:
        if (_TOTAL_WAVENUMER_DIM in da.dims) & (_LONGITUDINAL_WAVENUMER_DIM in da.dims):
            da = repack_mn(da, da.sizes[_TOTAL_WAVENUMER_DIM]-1)
        elif _HARMONIC_DIM not in da.dims:
            raise ValueError('Unable to identify harmonic dimension(s)')
        da = prep_for_inv_spharm(da)
    
    if n_lat is None:
        n_trunc = N_trunc(da.sizes[_HARMONIC_DIM])
        n_lat = 2 * (n_trunc + 1)
    n_lon = 2*n_lat
    
    st = create_spharmt(n_lon, n_lat, gridtype=gridtype)

    if _NON_HORIZONTAL_DIM in da.dims:
        output_core_dims = [[lat_name, lon_name, _NON_HORIZONTAL_DIM]]
        output_sizes = {lat_name: n_lat,
                        lon_name: n_lon,
                        _NON_HORIZONTAL_DIM: da.sizes[_NON_HORIZONTAL_DIM]}
        real = xr.apply_ufunc(_spectogrd, st, da,
                              input_core_dims=[[], da.dims],
                              output_core_dims=output_core_dims,
                              output_sizes=output_sizes,
                              exclude_dims=set((_HARMONIC_DIM, )), dask='allowed') \
                 .unstack(_NON_HORIZONTAL_DIM)
    else:
        output_core_dims = [[lat_name, lon_name]]
        output_sizes = {lat_name: n_lat, lon_name: n_lon}

        real = xr.apply_ufunc(_spectogrd, st, da,
                              input_core_dims=[[], da.dims],
                              output_core_dims=output_core_dims,
                              output_sizes=output_sizes,
                              exclude_dims=set((_HARMONIC_DIM, )), dask='allowed')
    
    lat, lon = get_spharm_grid(n_lat, gridtype)
    real[lat_name] = lat
    real[lon_name] = lon
    real = add_attrs(real, **{'xspharm_history':'spectogrd(..'
                              ' gridtype=' + gridtype + 
                              ', n_lat=' + str(n_lat) +
                              ', prepped=' + str(prepped) +')'})
    return real


# ===================================================================================================
# getpsichi
# ===================================================================================================

def getpsichi(u_grid, v_grid, gridtype, lat_dim='lat', lon_dim='lon', n_trunc=None):
    """
        Returns streamfunction (psi) and velocity potential (chi) using spharm package
        
        Parameters
        ----------
        u_grid : xarray DataArray
            Array containing grid of zonal winds
        v_grid : xarray DataArray
            Array containing grid of meridional winds
        gridtype : "gaussian" or "regular"
            Grid type of da
        n_trunc : int, optional
            Spectral truncation limit
            
        Returns
        -------
        xarray Dataset
            Arrays containing the streamfunction and velocity potential
    """

    def _getpsichi(st, u, v, n_trunc):
        """
            Wrap Spharmt.getpsichi to be dask compatible
        """
        if isinstance(u, darray.core.Array):
            @darray.as_gufunc(signature="(),(),()->(),()", 
                              output_dtypes=(float, float),
                              allow_rechunk=True)
            def _gu_getpsichi(u, v, n_trunc):
                return st.getpsichi(u, v, n_trunc)
            psi, chi = _gu_getpsichi(u, v, n_trunc)
            return psi, chi
        else:
            psi, chi = st.getpsichi(u, v, n_trunc)
            return psi, chi
    
    if n_trunc is None:
        n_trunc = int(u_grid.sizes[lat_dim]/2) - 1
        
    u_grid, _ = prep_for_spharm(u_grid)
    v_grid, flipped = prep_for_spharm(v_grid)
    
    st = create_spharmt(u_grid.sizes[lon_dim], u_grid.sizes[lat_dim], gridtype=gridtype)

    psi, chi = xr.apply_ufunc(_getpsichi, st, u_grid, v_grid, n_trunc,
                                input_core_dims=[[], u_grid.dims, v_grid.dims, []],
                                output_core_dims=[u_grid.dims, v_grid.dims],
                                dask='allowed')

    psichi = xr.merge([psi.unstack(_NON_HORIZONTAL_DIM).rename('psi'), 
                       chi.unstack(_NON_HORIZONTAL_DIM).rename('chi')])
        
    return add_attrs(psichi, **{'xspharm_history':'getpsichi(..'
                                ' gridtype=' + gridtype + 
                                ', n_trunc=' + str(n_trunc) +')'})


# Tested, and takes same amount of time to do separately with map_blocks, i.e.:
# def getpsi(u_grid, v_grid, gridtype, n_trunc=None):
#     """
#         Returns stream function (psi) using spharm package
        
#         Parameters
#         ----------
#         u_grid : xarray DataArray
#             Array containing grid of zonal winds
#         v_grid : xarray DataArray
#             Array containing grid of meridional winds
#         gridtype : "gaussian" or "regular"
#             Grid type of da
#         n_trunc : int, optional
#             Spectral truncation limit
            
#         Returns
#         -------
#         xarray DataArray
#             Arrays containing the stream function
#     """

#     def _getpsi(st, u, v, n_trunc):
#         """
#             Wrap Spharmt.getpsichi to be dask compatible
#         """
#         def _ggetpsi(st, u, v, n_trunc):
#             psi, _ = st.getpsichi(u, v, n_trunc)
#             return psi
        
#         if isinstance(u, darray.core.Array):
#             return darray.map_blocks(_ggetpsi, st, u, v, n_trunc,
#                                      dtype=np.float)
#         else:
#             return _ggetpsi(st, u, v, n_trunc)
 
#     if n_trunc is None:
#         n_trunc = int(u_grid.sizes[lat_dim]/2) - 1
        
#     u_grid, _ = prep_for_spharm(u_grid)
#     v_grid, flipped = prep_for_spharm(v_grid)
    
#     st = create_spharmt(u_grid.sizes[lon_dim], u_grid.sizes[lat_dim], gridtype=gridtype)

#     psi = xr.apply_ufunc(_getpsi, st, u_grid, v_grid, n_trunc,
#                          input_core_dims=[[], u_grid.dims, v_grid.dims, []],
#                          output_core_dims=[u_grid.dims],
#                          dask='allowed').unstack(_NON_HORIZONTAL_DIM).rename('psi')
        
#     return add_attrs(psi, **{'xspharm_history':'getpsi(..'
#                              ' gridtype=' + gridtype + 
#                              ', n_trunc=' + str(n_trunc) +')'})
# def getchi(u_grid, v_grid, gridtype, n_trunc=None):
#     """
#         Returns velocity potential (chi) using spharm package
        
#         Parameters
#         ----------
#         u_grid : xarray DataArray
#             Array containing grid of zonal winds
#         v_grid : xarray DataArray
#             Array containing grid of meridional winds
#         gridtype : "gaussian" or "regular"
#             Grid type of da
#         n_trunc : int, optional
#             Spectral truncation limit
            
#         Returns
#         -------
#         xarray DataArray
#             Arrays containing the stream function
#     """

#     def _getchi(st, u, v, n_trunc):
#         """
#             Wrap Spharmt.getpsichi to be dask compatible
#         """
#         def _ggetchi(st, u, v, n_trunc):
#             _, chi = st.getpsichi(u, v, n_trunc)
#             return chi
        
#         if isinstance(u, darray.core.Array):
#             return darray.map_blocks(_ggetchi, st, u, v, n_trunc,
#                                      dtype=np.float)
#         else:
#             return _ggetchi(st, u, v, n_trunc)

#     if n_trunc is None:
#         n_trunc = int(u_grid.sizes[lat_dim]/2) - 1
        
#     u_grid, _ = prep_for_spharm(u_grid)
#     v_grid, flipped = prep_for_spharm(v_grid)
    
#     st = create_spharmt(u_grid.sizes[lon_dim], u_grid.sizes[lat_dim], gridtype=gridtype)

#     chi = xr.apply_ufunc(_getchi, st, u_grid, v_grid, n_trunc,
#                          input_core_dims=[[], u_grid.dims, v_grid.dims, []],
#                          output_core_dims=[u_grid.dims],
#                          dask='allowed').unstack(_NON_HORIZONTAL_DIM).rename('chi')
        
#     return add_attrs(chi, **{'xspharm_history':'getchi(..'
#                              ' gridtype=' + gridtype + 
#                              ', n_trunc=' + str(n_trunc) +')'})