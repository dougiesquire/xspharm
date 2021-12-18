import spharm

import numpy as np
import xarray as xr

import dask
import dask.array as dsa


def empty_dask_array(shape, dtype=float, chunks=None):
    """A dask array that errors if you try to compute it
    Stolen from https://github.com/xgcm/xhistogram/blob/master/xhistogram/test/fixtures.py
    """

    def raise_if_computed():
        raise ValueError("Triggered forbidden computation on dask array")

    a = dsa.from_delayed(dask.delayed(raise_if_computed)(), shape, dtype)
    if chunks is not None:
        a = a.rechunk(chunks)
    return a


def example_da(gridtype, n_additional_dims, wrap="numpy"):
    """An example DataArray.
        The first two dimensions are lat and lon and data are \
        replicated along additional dims
    """
    n_lat = 90
    n_additional_dim = 2

    if gridtype == "gaussian":
        lat = spharm.gaussian_lats_wts(n_lat)[0]
        lon = np.linspace(0, 360, 2 * len(lat) + 1)[0:-1]
    elif gridtype == "regular":
        if n_lat % 2 == 0:
            lat = np.linspace(90 - 90 / n_lat, -90 + 90 / n_lat, n_lat)
        else:
            lat = np.linspace(90, -90, n_lat)
        lon = np.linspace(0, 360, 2 * len(lat) + 1)[0:-1]

    coords = [lat, lon] + [range(n_additional_dim) for _ in range(n_additional_dims)]
    dims = ["lat", "lon"] + [f"dim_{i}" for i in range(n_additional_dims)]
    shape = (len(lat), len(lon), *(n_additional_dim,) * n_additional_dims)

    if wrap == "dask_nocompute":
        data = empty_dask_array(shape)
    else:
        data = np.expand_dims(
            np.random.random(size=shape[:2]), list(range(2, len(shape)))
        )
        data = np.tile(data, (*shape[2:],))
        if wrap == "dask":
            data = dsa.from_array(data)
    return xr.DataArray(data, coords=coords, dims=dims)
