# xspharm

[![tests](https://github.com/dougiesquire/xspharm/actions/workflows/tests.yml/badge.svg)](https://github.com/dougiesquire/xspharm/actions/workflows/tests.yml)
[![pre-commit](https://github.com/dougiesquire/xspharm/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/dougiesquire/xspharm/actions/workflows/pre-commit.yml)
[![codecov](https://codecov.io/gh/dougiesquire/xspharm/branch/master/graph/badge.svg?token=XPK4V5X1TH)](https://codecov.io/gh/dougiesquire/xspharm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/dougiesquire/xspharm/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

A simple dask-enabled xarray wrapper for `pyspharm` (which wraps SPHEREPACK)

Taken, adapted and extended from [spencerclark](https://github.com/spencerkclark)'s [gist](https://gist.github.com/spencerkclark/6a8e05a492111e52d8d8fb407d332611)

### Installation

This package is not on PyPI. To install:
```
# Install/activate dependencies or activate an environment with xarray and dask
conda env create -f environment.yml
conda activate xspharm

pip install .
```

## Quick overview of (complex) spherical harmonics
In two dimensional space, the Fourier series can be arrived at by considering polymonials on a circle that are harmonic (ie the two-dimensional Laplacian of the polynomial equal to zero), homogenous (multiplicative scaling behaviour). The approach leads to trigonometric polynomials as the basis functions (i.e. linear combinations of two elements $\left[e^{im\theta}, e^{-im\theta}\right]$).

In a equivalent manner, Spherical Harmonics can be arrived at by considering homogeneous harmonic polynomials on a sphere. Separating the polynomials to comprise trigonometric polynomials in longitude (lines of fixed latitude are circles), leads to a set of Associated Legendre functions, $P_{n}^{m} \left(\cos{\phi}\right)$, of order $m$ and degree $n$, for each $m$. A disadvantage of this organisation is that it makes the poles into special points, but separating the variables is so useful that there are no competitive options.

That is, any real square-integrable function can be expressed as a series of spherical harmonic functions,
\begin{equation}
f\left(\phi,\theta\right) = \sum_{n=0}^{\infty} \sum_{m=-n}^{n} f_{n}^{m} Y_{n}^{m} \left(\phi,\theta\right) = \sum_{n=0}^{\infty} \sum_{m=-n}^{n} f_{n}^{m} \hat{P}_{n}^{m} \left(\cos{\phi}\right) e^{im\theta}.
\end{equation}
(Note, there is a 1/$\sqrt{ 2\pi }$ that can appear in each of the various terms or outside the sum, depending on the derivation - see below and [1]). In the above, $f_{n}^{m}$ are the complex spectral harmonic coefficients of $f\left(\phi,\theta\right)$; $Y_{n}^{m}$ are the corresponding complex spherical harmonic functions; $\hat{P}_{n}^{m} \left(\cos{\phi}\right)$ are the normalised Associated Legendre functions; $m$ and $n$ are the spherical harmonic order and degree, respectively; $\phi$ and $\theta$ are the co-latitude and longitude, respectively.

The complex spherical harmonic functions possess a symmetry relationship for positive and negative angular orders,
\begin{equation}
Y_{n}^{m*} = \left(-1\right)^{m} Y_{n}^{-m}.
\end{equation}
In addition, if the function, $f\left(\phi,\theta\right)$, is entirely real, then the real and complex spherical harmonic coefficients are related by,
\begin{equation}
f_{n}^{m} =
    \begin{cases}
        \left( f_{nm} - i f_{n-m} \right) & \text{if } m > 0 \\
        f_{l0}                            & \text{if } m = 0 \\
        \left( -1 \right)^{m} f_{n}^{-m*} & \text{if } m < 0, \\
    \end{cases}
\end{equation}
where the use of only subscripts indicates the values from a real spectral harmonic analysis. Thus, it is not entirely necessary to save all coefficients - for example, I think `spharm` provides only coefficients for $m>0$.

### Discretizing the problem: quadrature
A quadrature is a rule for converting an integral into a sum:
\begin{equation}
\int_{-a}^{b} f \left(x\right) dx = \sum_{j=1}^{n} w_{j} f \left(x_j\right).
\end{equation}
For Legendre polynomials, Gaussian quadrature provides an exact quadrature for polynomials of degree less than 2N using only N points. One can also use true equispaced nodes in latitude, and these correspond to Chebychev nodes in x. The Chebychev nodes have several nice properties, but require twice as many points as the gaussian nodes. For trigonometric polynomials, the appropriate quadrature weights are all 1 and the quadrature points are an equispaced sampling. Thus, when evaulating Spherical Harmonics it is common to use N points in latitude with Gaussian spacing and weighting, and 2N points in longitude with equal spacing.

### Interpretation of spherical harmonics
Parseval’s theorem in Cartesian geometry relates the integral of a function squared to the sum of the squares of the function’s Fourier coefficients. This relation is easily extended to spherical geometry using the orthogonality properties of the spherical harmonic functions. Defining power to be the integral of the function squared divided by the area it spans, the total power of a function is equal to a sum over its power spectrum,
\begin{equation}
\frac{1}{4\pi}\int_{\Omega} |f|^2 \left(\phi, \theta\right) d\Omega = \sum_{n=0}^{\infty} S_{ff} \left(n\right),
\end{equation}
where $d\Omega$ is the differential surface area on the unit sphere (for $0 \leq \theta \leq 360$ and $0 \leq \phi \leq 180$, $d\Omega = \sin\phi d\phi d\theta$). For the most common form of spherical harmonic normalisation, $4\pi-normalisation$, the power spectrum, $S$, is related to the spectral harmonic coefficients by,
\begin{equation}
S_{ff}\left(n\right) = \sum_{m=-n}^{n} |f_{n}^{m}|^2.
\end{equation}
See reference [2] for other types of normalisation. If the function $f\left(\phi,\theta\right)$ has a zero mean, $S_{ff}\left(n\right)$ represents the contribution to the variance as a function of degree $n$. 

### `shparm`
The `spharm` package is a wrapper on UCAR's FORTRAN77 library `SPHEREPACK`. There is documentation on the latter [3] which notes the use of normalized Associated Legendre functions of the form,
\begin{equation}
\hat{P}_{n}^{m} = \sqrt{ \frac{2n + 1}{2} \frac{\left(n-m\right)!}{\left(n+m\right)!} } P_{n}^{m},
\end{equation}
whereas the typical $4\pi-normalized$ harmonics use,
\begin{equation}
\hat{P}_{n}^{m} = \sqrt{ \left(2n + 1\right) \frac{\left(n-m\right)!}{\left(n+m\right)!} } P_{n}^{m}.
\end{equation}
Thus, to convert to $4\pi-normalized$ harmonics, `spharm`/`SPHEREPACK` coefficients should be normalised by 1/$\sqrt{ 2 }$.

Additionally, the spherical harmonic decomposition in `SPHEREPACK` is defined as,
\begin{equation}
f\left(\phi,\theta\right) = \sum_{n=0}^{\infty} {\sum_{m=0}^{n}}^{'} f_{n}^{m} P_{n}^{m} e^{im\theta},
\end{equation}
where the prime notation on the sum indicates that the fist term corresponding to $m=0$ is multiplied by $1/2$. That is, `spharm` returns coefficients only for $m > 0$, where,
\begin{equation}
S_{ff}\left(n\right) = \left|\frac{f_{n}^{0}}{\sqrt{2}}\right|^2 + \sum_{m=1}^{n} 2\left|\frac{f_{n}^{m}}{\sqrt{2}}\right|^2.
\end{equation}

### References
[1] Nice overview: https://pdfs.semanticscholar.org/fcc6/5f4b2c626fb0b9685999d16a8b42799cd15b.pdf 

[2] `SHTools`: https://shtools.oca.eu/shtools/complex-spherical-harmonics.html and https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018GC007529

[3] `SPHEREPACK`: https://www2.cisl.ucar.edu/resources/legacy/spherepack/documentation
