import numpy as np
import xarray as xr
import dask.array as dsa

from .utils import (
    _HARMONIC_DIM,
    _TOTAL_WAVENUMER_DIM,
    _LONGITUDINAL_WAVENUMER_DIM,
    _NON_HORIZONTAL_DIM,
    _N_harmonics,
    _create_spharmt,
    _add_attrs,
    _prep_for_spharm,
    _prep_for_inv_spharm,
    get_spharm_grid,
    repack_mn,
    unpack_mn,
)


def grdtospec(
    da,
    gridtype,
    lat_dim="lat",
    lon_dim="lon",
    n_trunc=None,
    unpack_wavenumber_pairs=False,
    prepped=False,
):
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
        if isinstance(da, dsa.core.Array):
            if da.ndim == 3:
                chunks = ((_N_harmonics(n_trunc),), da.chunks[-1])
            else:
                chunks = (_N_harmonics(n_trunc),)
            return dsa.map_blocks(
                st.grdtospec,
                da,
                n_trunc,
                chunks=chunks,
                dtype=np.complex,
                drop_axis=(0, 1),
                new_axis=(0,),
            )
        else:
            return st.grdtospec(da)

    if not prepped:
        da, flipped = _prep_for_spharm(da, lat_dim=lat_dim, lon_dim=lon_dim)

    if n_trunc is None:
        n_trunc = int(da.sizes[lat_dim] / 2) - 1

    st = _create_spharmt(da.sizes[lon_dim], da.sizes[lat_dim], gridtype=gridtype)

    if _NON_HORIZONTAL_DIM in da.dims:
        output_core_dims = [[_HARMONIC_DIM, _NON_HORIZONTAL_DIM]]
        output_sizes = {
            _HARMONIC_DIM: _N_harmonics(n_trunc),
            _NON_HORIZONTAL_DIM: da.sizes[_NON_HORIZONTAL_DIM],
        }
        coeffs = xr.apply_ufunc(
            _grdtospec,
            st,
            da,
            n_trunc,
            input_core_dims=[[], da.dims, []],
            output_core_dims=output_core_dims,
            output_sizes=output_sizes,
            exclude_dims=set((lat_dim, lon_dim)),
            dask="allowed",
        ).unstack(_NON_HORIZONTAL_DIM)
    else:
        output_core_dims = [[_HARMONIC_DIM]]
        output_sizes = {_HARMONIC_DIM: _N_harmonics(n_trunc)}

        coeffs = xr.apply_ufunc(
            _grdtospec,
            st,
            da,
            n_trunc,
            input_core_dims=[[], da.dims, []],
            output_core_dims=output_core_dims,
            output_sizes=output_sizes,
            exclude_dims=set((lat_dim, lon_dim)),
            dask="allowed",
        )

    coeffs[_HARMONIC_DIM] = range(coeffs.sizes[_HARMONIC_DIM])
    coeffs = _add_attrs(
        coeffs,
        **{
            "xspharm_history": "grdtospec(.."
            " gridtype="
            + gridtype
            + ", n_trunc="
            + str(n_trunc)
            + ", unpack_wavenumber_pairs="
            + str(unpack_wavenumber_pairs)
            + ", prepped="
            + str(prepped)
            + ")"
        }
    )

    if unpack_wavenumber_pairs:
        return unpack_mn(coeffs, n_trunc)
    else:
        return coeffs


def spectogrd(
    da, gridtype="gaussian", n_lat=None, prepped=False, lat_name="lat", lon_name="lon"
):
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
        return int(np.sqrt(2 * n_harmonics) - 1)

    def _spectogrd(st, da):
        """
        Transform a variable from spectral to grid space
        """
        if isinstance(da, dsa.core.Array):
            if da.ndim == 2:
                chunks = ((st.nlat,), (st.nlon,), da.chunks[-1])
            else:
                chunks = ((st.nlat,), (st.nlon,))
            return dsa.map_blocks(
                st.spectogrd, da, chunks=chunks, drop_axis=(0,), new_axis=(0, 1)
            )
        else:
            return st.spectogrd(da)

    if not prepped:
        if (_TOTAL_WAVENUMER_DIM in da.dims) & (_LONGITUDINAL_WAVENUMER_DIM in da.dims):
            da = repack_mn(da, da.sizes[_TOTAL_WAVENUMER_DIM] - 1)
        elif _HARMONIC_DIM not in da.dims:
            raise ValueError("Unable to identify harmonic dimension(s)")
        da = _prep_for_inv_spharm(da)

    if n_lat is None:
        n_trunc = N_trunc(da.sizes[_HARMONIC_DIM])
        n_lat = 2 * (n_trunc + 1)
    n_lon = 2 * n_lat

    st = _create_spharmt(n_lon, n_lat, gridtype=gridtype)

    if _NON_HORIZONTAL_DIM in da.dims:
        output_core_dims = [[lat_name, lon_name, _NON_HORIZONTAL_DIM]]
        output_sizes = {
            lat_name: n_lat,
            lon_name: n_lon,
            _NON_HORIZONTAL_DIM: da.sizes[_NON_HORIZONTAL_DIM],
        }
        real = xr.apply_ufunc(
            _spectogrd,
            st,
            da,
            input_core_dims=[[], da.dims],
            output_core_dims=output_core_dims,
            output_sizes=output_sizes,
            exclude_dims=set((_HARMONIC_DIM,)),
            dask="allowed",
        ).unstack(_NON_HORIZONTAL_DIM)
    else:
        output_core_dims = [[lat_name, lon_name]]
        output_sizes = {lat_name: n_lat, lon_name: n_lon}

        real = xr.apply_ufunc(
            _spectogrd,
            st,
            da,
            input_core_dims=[[], da.dims],
            output_core_dims=output_core_dims,
            output_sizes=output_sizes,
            exclude_dims=set((_HARMONIC_DIM,)),
            dask="allowed",
        )

    lat, lon = get_spharm_grid(n_lat, gridtype)
    real[lat_name] = lat
    real[lon_name] = lon
    real = _add_attrs(
        real,
        **{
            "xspharm_history": "spectogrd(.."
            " gridtype="
            + gridtype
            + ", n_lat="
            + str(n_lat)
            + ", prepped="
            + str(prepped)
            + ")"
        }
    )
    return real


def getpsichi(u_grid, v_grid, gridtype, lat_dim="lat", lon_dim="lon", n_trunc=None):
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
        if isinstance(u, dsa.core.Array):

            @dsa.as_gufunc(
                signature="(),(),()->(),()",
                output_dtypes=(float, float),
                allow_rechunk=True,
            )
            def _gu_getpsichi(u, v, n_trunc):
                return st.getpsichi(u, v, n_trunc)

            psi, chi = _gu_getpsichi(u, v, n_trunc)
            return psi, chi
        else:
            psi, chi = st.getpsichi(u, v, n_trunc)
            return psi, chi

    if n_trunc is None:
        n_trunc = int(u_grid.sizes[lat_dim] / 2) - 1

    u_grid, _ = _prep_for_spharm(u_grid, lat_dim=lat_dim, lon_dim=lon_dim)
    v_grid, flipped = _prep_for_spharm(v_grid, lat_dim=lat_dim, lon_dim=lon_dim)

    st = _create_spharmt(
        u_grid.sizes[lon_dim], u_grid.sizes[lat_dim], gridtype=gridtype
    )

    psi, chi = xr.apply_ufunc(
        _getpsichi,
        st,
        u_grid,
        v_grid,
        n_trunc,
        input_core_dims=[[], u_grid.dims, v_grid.dims, []],
        output_core_dims=[u_grid.dims, v_grid.dims],
        dask="allowed",
    )

    psichi = xr.merge(
        [
            psi.unstack(_NON_HORIZONTAL_DIM).rename("psi"),
            chi.unstack(_NON_HORIZONTAL_DIM).rename("chi"),
        ]
    )

    return _add_attrs(
        psichi,
        **{
            "xspharm_history": "getpsichi(.."
            " gridtype=" + gridtype + ", n_trunc=" + str(n_trunc) + ")"
        }
    )
