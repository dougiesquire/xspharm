import pytest

import xspharm

from spharm import Spharmt

import numpy as np
import numpy.testing as npt

from .fixtures import example_da


@pytest.mark.parametrize("gridtype", ["regular", "gaussian"])
@pytest.mark.parametrize("n_broadcast_dims", [0, 1, 2])
@pytest.mark.parametrize("wrap", ["numpy", "dask", "dask_nocompute"])
@pytest.mark.parametrize("n_trunc", [None, 10])
def test_grdtospec(gridtype, n_broadcast_dims, wrap, n_trunc):
    """Test grdtospec relative to pyspharm"""
    data = example_da(gridtype, n_broadcast_dims, wrap)
    n_lat = data.sizes["lat"]
    n_lon = data.sizes["lon"]

    res = xspharm.grdtospec(data, gridtype, n_trunc)

    if wrap != "dask_nocompute":
        data_2D = np.reshape(data.values, (n_lat, n_lon, -1))[:, :, 0]
        if n_trunc is None:
            n_trunc = int((n_lat / 2) - 1)
        st = Spharmt(n_lon, n_lat, xspharm.utils.EARTH_RADIUS, gridtype)
        ver = st.grdtospec(data_2D, n_trunc)

        npt.assert_allclose(res.T - ver, 0.0)


@pytest.mark.parametrize("gridtype", ["regular", "gaussian"])
@pytest.mark.parametrize("n_broadcast_dims", [0, 1, 2])
@pytest.mark.parametrize("wrap", ["numpy", "dask", "dask_nocompute"])
@pytest.mark.parametrize("n_trunc", [None, 10])
@pytest.mark.parametrize("unpack", [True, False])
def test_spectogrd(gridtype, n_broadcast_dims, wrap, n_trunc, unpack):
    """Test grdtospec-spectogrd roundtrip"""
    data = example_da(gridtype, n_broadcast_dims, wrap)
    n_lat = data.sizes["lat"]
    n_lon = data.sizes["lon"]

    coeff = xspharm.grdtospec(data, gridtype, n_trunc, unpack)

    res = xspharm.spectogrd(coeff, gridtype, n_lat)

    if wrap != "dask_nocompute":
        if unpack:
            if n_trunc is None:
                n_trunc = int((n_lat / 2) - 1)
            coeff = xspharm.utils.repack_mn(coeff, n_trunc)
        coeff_1D = np.reshape(
            coeff.values, (coeff.sizes[xspharm.utils._HARMONIC_DIM], -1)
        )[:, 0]
        st = Spharmt(n_lon, n_lat, xspharm.utils.EARTH_RADIUS, gridtype)
        ver = st.spectogrd(coeff_1D)

        npt.assert_allclose(res.stack(n=["lat", "lon"]) - ver.flatten(), 0.0)


@pytest.mark.parametrize("gridtype", ["regular", "gaussian"])
@pytest.mark.parametrize("n_broadcast_dims", [0, 1, 2])
@pytest.mark.parametrize("wrap", ["numpy", "dask", "dask_nocompute"])
@pytest.mark.parametrize("n_trunc", [None, 10])
def test_getpsichi(gridtype, n_broadcast_dims, wrap, n_trunc):
    """Test getpsichi relative to pyspharm"""
    u = example_da(gridtype, n_broadcast_dims, wrap)
    v = example_da(gridtype, n_broadcast_dims, wrap)
    n_lat = u.sizes["lat"]
    n_lon = u.sizes["lon"]

    res = xspharm.getpsichi(u, v, gridtype, n_trunc)

    if wrap != "dask_nocompute":
        u_2D = np.reshape(u.values, (n_lat, n_lon, -1))[:, :, 0]
        v_2D = np.reshape(v.values, (n_lat, n_lon, -1))[:, :, 0]
        if n_trunc is None:
            n_trunc = int((n_lat / 2) - 1)
        st = Spharmt(n_lon, n_lat, xspharm.utils.EARTH_RADIUS, gridtype)
        ver_psi, ver_chi = st.getpsichi(u_2D, v_2D, n_trunc)

        npt.assert_allclose(res["psi"].stack(n=["lat", "lon"]) - ver_psi.flatten(), 0.0)
        npt.assert_allclose(res["chi"].stack(n=["lat", "lon"]) - ver_chi.flatten(), 0.0)