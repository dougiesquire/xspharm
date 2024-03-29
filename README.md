## xspharm

[![tests](https://github.com/dougiesquire/xspharm/actions/workflows/tests.yml/badge.svg)](https://github.com/dougiesquire/xspharm/actions/workflows/tests.yml)
[![pre-commit](https://github.com/dougiesquire/xspharm/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/dougiesquire/xspharm/actions/workflows/pre-commit.yml)
[![codecov](https://codecov.io/gh/dougiesquire/xspharm/branch/master/graph/badge.svg?token=XPK4V5X1TH)](https://codecov.io/gh/dougiesquire/xspharm)
[![License:
MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/dougiesquire/xspharm/blob/master/LICENSE)
[![Code style:
black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

A simple dask-enabled xarray wrapper for `pyspharm` (which wraps
SPHEREPACK).

Taken, adapted and extended from
[spencerclark](https://github.com/spencerkclark)’s
[gist](https://gist.github.com/spencerkclark/6a8e05a492111e52d8d8fb407d332611).

See the start of [this notebook](https://github.com/dougiesquire/xspharm/blob/master/notebooks/pyspharm.ipynb) for an intro/overview of spherical harmonics.

## Installation

This package is not on PyPI. To install:

```
    # Install/activate dependencies or activate an environment with xarray and dask
    conda env create -f environment.yml
    conda activate xspharm

    pip install .
```
