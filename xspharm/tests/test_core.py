import pytest

import xspharm

from spharm import Spharmt

import numpy as np
import numpy.testing as npt

from .fixtures import example_da


@pytest.mark.parametrize("gridtype", ["regular", "gaussian"])
@pytest.mark.parametrize("n_broadcast_dims", [0, 1, 2])
@pytest.mark.parametrize("wrap", ["numpy", "dask", "dask_nocompute"])
@pytest.mark.parametrize("n_trunc", [None, 44, 20, 10])
def test_grdtospec(gridtype, n_broadcast_dims, wrap, n_trunc):
    """Test grdtospec relative to pyspharm"""
    data = example_da(gridtype, n_broadcast_dims, wrap)

    res = xspharm.grdtospec(data, gridtype, n_trunc)

    if wrap != "dask_nocompute":
        data_2D = np.reshape(data.values, (data.sizes["lat"], data.sizes["lon"], -1))[
            :, :, 0
        ]
        if n_trunc is None:
            n_trunc = (data.sizes["lat"] / 2) - 1
        st = Spharmt(
            data.sizes["lon"], data.sizes["lat"], xspharm.utils.EARTH_RADIUS, gridtype
        )
        ver = st.grdtospec(data_2D, n_trunc)

        npt.assert_allclose(res.T - ver, 0.0)
