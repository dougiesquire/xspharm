import numpy as np
import xarray as xr


def example_da(gridtype, n_additional_dims, dask):
    """An example DataArray.
        The first two dimensions are lat and lon and data are \
        replicated along additional dims
    """
    n_additional_dim = 2

    if gridtype == "gaussian":
        data_2D = xr.open_dataset("./xspharm/test_data/example_data_gaussian.nc")[
            "t_surf"
        ]
    elif gridtype == "regular":
        data_2D = xr.open_dataset("./xspharm/test_data/example_data_regular.nc")[
            "t_surf"
        ]

    if n_additional_dims == 0:
        return data_2D
    else:
        additional_dims = xr.DataArray(
            np.empty((n_additional_dim,) * n_additional_dims),
            coords={
                f"dim_{i}": range(n_additional_dim) for i in range(n_additional_dims)
            },
        )
        data = xr.broadcast(data_2D, additional_dims)[0]

        if dask:
            data = data.chunk()

        return data
