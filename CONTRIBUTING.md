## Contributor guide

This package is still is its very early stages of development. The following covers some general guidelines for maintainers and contributors.

#### Preparing Pull Requests
1. Fork this respository. It's fine to use "xspharm" as your fork repository name because it will live under your username.

2. Clone your fork locally, connect your repository to the upstream (main project), and create a branch to work on:

```
$ git clone git@github.com:YOUR_GITHUB_USERNAME/xspharm.git
$ cd xspharm
$ git remote add upstream git@github.com:dougiesquire/xspharm.git
$ git checkout -b YOUR-BUGFIX-FEATURE-BRANCH-NAME master
```

3. Install `xspharm`'s dependencies into a new conda environment:

```
$ conda env create -f environment.yml
$ conda activate xspharm
```

Aside: it is handy to install your conda environment as an ipykernel. This makes a kernel with the `xspharm` environment available from within Jupyter and you won't have to restart Jupyter to effectuate any changes/updates you make to the environment (simply restarting the kernel will do):

```
python -m ipykernel install --user --name xspharm --display-name "Python (xspharm)"
```
 
4. Install `xspharm` using the editable flag (meaning any changes you make to the package will be reflected directly in your environment):

```
$ pip install --no-deps -e .
```

Aside: to have the changes you make to the package register immediately when running IPython (e.g. a Jupyter notebook), run the following magic commands:

```
%load_ext autoreload
%autoreload 2 
```

5. This project uses `black` to format code and `flake8` for linting. We use `pre-commit` to ensure these have been run. Please set up commit hooks by running the following. This will mean that `black` and `flake8` are run whenever you make a commit:

```
pre-commit install
```

You can also run `pre-commit` manually at any point to format your code:

```
pre-commit run --all-files
 ```

6. Start making and committing your edits, including adding docstrings to functions and tests to `xspharm/tests` to check that your contributions are doing what they're suppose to. Please try to follow [numpydoc style](https://numpydoc.readthedocs.io/en/latest/format.html) for docstrings. To run the test suite:

```
pytest xspharm
```
