from setuptools import find_packages, setup
import versioneer

setup(
    name="xspharm",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Dougie Squire",
    url="https://github.com/dougiesquire/xspharm",
    description="A simple dask-enabled xarray wrapper for `pyspharm`",
    long_description="A simple dask-enabled xarray wrapper for `pyspharm` (which wraps SPHEREPACK)",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "xarray",
        "dask",
        "pyspharm",
    ],
)
