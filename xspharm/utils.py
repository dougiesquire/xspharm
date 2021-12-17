"""Utility functions for xspharm"""

import numpy as np
import xarray as xr

import dask.array as dsa

import spharm
from spharm import Spharmt


_HARMONIC_DIM = "harmonic"
_TOTAL_WAVENUMER_DIM = "n"
_LONGITUDINAL_WAVENUMER_DIM = "m"
_NON_HORIZONTAL_DIM = "non_horizontal"
EARTH_RADIUS = 6370997.0


def _wraps_dask_array(da):
    """
    Check if an xarray object wraps a dask array
    """
    return isinstance(da.data, dsa.core.Array)


def _get_other_dims(da, dim_exclude):
    """
    Returns all dimensions in provided xarray object excluding dim_exclude
    """

    dims = da.dims

    if dim_exclude is None:
        return dims
    else:
        if isinstance(dims, str):
            dims = [dims]
        if isinstance(dim_exclude, str):
            dim_exclude = [dim_exclude]

        return tuple(set(dims) - set(dim_exclude))


def _flip_lat(da, lat_name):
    """
    Flip latitude dimension
    """
    return da.isel(**{lat_name: slice(None, None, -1)})


def _create_spharmt(n_lon, n_lat, gridtype):
    """
    Initialise Spharmt object
    """
    return Spharmt(n_lon, n_lat, rsphere=EARTH_RADIUS, gridtype=gridtype)


def _stack_non_horizontal_dims(da, non_horizontal_dims):
    """
    If present, stack all non-horizontal dims onto one dimension
    """
    dims_to_stack = _get_other_dims(da, non_horizontal_dims)
    if dims_to_stack:
        da = da.stack(**{_NON_HORIZONTAL_DIM: dims_to_stack})
    return da


def _N_harmonics(n_trunc):
    """
    Return the number of harmonics
    """
    return (n_trunc + 1) * (n_trunc + 2) // 2


def _order_dims_first(da, first_dims):
    """
    Order dims such that first_dims come first
    """
    order = first_dims + _get_other_dims(da, first_dims)
    return da.transpose(*order)


def _make_single_chunk(da, dims):
    """
    If underlying data are chunked, rechunk specified dims to single chunk
    """
    chunks = {dim: -1 for dim in dims}

    if _wraps_dask_array(da):
        da = da.chunk(chunks)
    return da


def _add_attrs(da, **kwargs):
    """
    Add attributes to xarray object
    """
    for key, value in kwargs.items():
        if key in da.attrs:
            da.attrs[key] = da.attrs[key] + value
        else:
            da.attrs[key] = value
    return da


def _prep_for_spharm(da, lat_dim="lat", lon_dim="lon"):
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
            Array containing data that has been prepared for use with \
            spharm, and a  boolean indicating whether the data was \
            latitudinally flipped
    """

    def _orient_latitude_north_south(da, lat_dim):
        """
        Orients data such that northern latitudes come first
        Returns the transformed array as well as flag noting if the \
        data were flipped.
        """
        if all(da[lat_dim].diff(lat_dim) > 0.0):
            return _flip_lat(da, lat_dim), True
        else:
            return da, False

    da = _stack_non_horizontal_dims(da, (lat_dim, lon_dim))
    da = _order_dims_first(da, (lat_dim, lon_dim))
    da = _make_single_chunk(da, (lat_dim, lon_dim))
    return _orient_latitude_north_south(da, lat_dim)


def _prep_for_inv_spharm(da):
    """
        Prepare DataArray for use with inverse spharm (e.g. spectogrd)

        Parameters
        ----------
        da : xarray DataArray
            Input DataArray

        Returns
        -------
        xr.DataArray, boolean
            Array containing data that has been prepared for use with \
            spharm, and a boolean indicating whether the data was \
            latitudinally flipped
    """

    da = _stack_non_horizontal_dims(da, (_HARMONIC_DIM,))
    da = _order_dims_first(da, (_HARMONIC_DIM,))
    return _make_single_chunk(da, (_HARMONIC_DIM,))


def get_spharm_grid(n_lat, gridtype):
    """
    Generate spharm latitude-longitude grid.

    Parameters
    ----------
    n_lat : int
        Number of latitude points
    gridtype : str
        Grid type to return. Options are "gaussian" or "regular"

    Returns
    -------
    lat : np.array
        Array of grid latitude points
    lon : np.array
        Array of grid longitude points
    """
    if gridtype == "gaussian":
        lat = spharm.gaussian_lats_wts(n_lat)[0]
        lon = np.linspace(0, 360, 2 * len(lat) + 1)[0:-1]
    elif gridtype == "regular":
        if n_lat % 2 == 0:
            lat = np.linspace(90 - 90 / n_lat, -90 + 90 / n_lat, n_lat)
        else:
            lat = np.linspace(90, -90, n_lat)
        lon = np.linspace(0, 360, 2 * len(lat) + 1)[0:-1]
    else:
        raise ValueError("Unrecognised gridtype")
    return lat, lon


def repack_mn(da, n_trunc):
    """
        Pack m-n wavenumber pairs into single dimension ordered as \
        expected by spharm.spectogrd

        Parameters
        ----------
        da : xr.DataArray
            Array containing _LONGITUDINAL_WAVENUMER_DIM and \
            _TOTAL_WAVENUMER_DIM to repack along single dimension
        n_trunc : int
            Spectral truncation limit of data in input array

        Returns
        -------
        xr.DataArray, boolean
            Array containing data unpacked along a new _HARMONIC_DIM \
            dimension
    """
    to_concat = []
    prev = 0
    for m in range(n_trunc + 1):
        da_h = (
            da.sel({_LONGITUDINAL_WAVENUMER_DIM: m}, drop=True)
            .sel({_TOTAL_WAVENUMER_DIM: slice(m, n_trunc + 1)})
            .rename({_TOTAL_WAVENUMER_DIM: _HARMONIC_DIM})
        )
        da_h[_HARMONIC_DIM] = range(prev, prev + n_trunc - m + 1)
        to_concat.append(da_h)
        prev += n_trunc - m + 1

    return xr.concat(to_concat, dim=_HARMONIC_DIM).chunk({_HARMONIC_DIM: -1})


def unpack_mn(da, n_trunc):
    """
        Unpack output from grdtospec into m-n wavenumber pairs

        Parameters
        ----------
        da : xr.DataArray
            Array containing _HARMONIC_DIM to be unpacked
        n_trunc : int
            Spectral truncation limit of data in input array

        Returns
        -------
        xr.DataArray, boolean
            Array containing data unpacked along a new dimensions \
            _LONGITUDINAL_WAVENUMER_DIM and _TOTAL_WAVENUMER_DIM
    """
    to_concat = []
    prev = 0
    for n in range(n_trunc + 1):
        da_n = da.isel({_HARMONIC_DIM: slice(prev, prev + n_trunc - n + 1)}).rename(
            {_HARMONIC_DIM: _TOTAL_WAVENUMER_DIM}
        )
        da_n[_TOTAL_WAVENUMER_DIM] = range(n, n_trunc + 1)
        to_concat.append(da_n)
        prev += n_trunc - n + 1

    unpacked = xr.concat(to_concat, dim=_LONGITUDINAL_WAVENUMER_DIM)
    unpacked[_LONGITUDINAL_WAVENUMER_DIM] = range(0, n_trunc + 1)
    return unpacked


def sum_along_m(coeffs):
    """
        Returns the sum along the longitudinal wavenumber dimension \
        of the provided coefficients computed using spharm, \
        accounting for the fact that spharm returns one side of the \
        decomposition

        Parameters
        ----------
        coeffs : xarray DataArray
            Array containing spherical harmonic coefficients

        Returns
        -------
        xr.DataArray
            Array containing the sum of the coefficients along the \
            longitudinal wavenumber
    """
    if _HARMONIC_DIM in coeffs.dims:
        raise ValueError(
            "Please first unpack the coefficients into m-n wavenumber pairs using unpack_mn"
        )
    return xr.concat(
        [coeffs.sel(m=0), 2 * coeffs.sel(m=slice(1, len(coeffs.m) + 1))], dim="m"
    ).sum("m")


def mean_along_m(coeffs):
    """
        Returns the mean along the longitudinal wavenumber dimension \
        of the provided coefficients computed using spharm, \
        accounting for the fact that spharm returns one side of the \
        decomposition

        Parameters
        ----------
        coeffs : xarray DataArray
            Array containing spherical harmonic coefficients

        Returns
        -------
        xr.DataArray
            Array containing the mean of the coefficients along the \
            longitudinal wavenumber
    """
    if _HARMONIC_DIM in coeffs.dims:
        raise ValueError(
            "Please first unpack the coefficients into m-n wavenumber pairs using unpack_mn"
        )
    return xr.concat(
        [coeffs.sel(m=0), 2 * coeffs.sel(m=slice(1, len(coeffs.m) + 1))], dim="m"
    ).mean("m")


def get_power(coeffs):
    """
        Returns the power, S(n), given the spherical harmonic \
        coefficients computed using xspharm

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
        raise ValueError(
            "Please first unpack the coefficients into m-n wavenumber pairs using unpack_mn"
        )
    return sum_along_m(abs(coeffs) ** 2)
