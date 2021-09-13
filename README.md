# ORB (TC) conformal example

This repo captures code and ideas for applying "Conformal Prediction for Simulation Models" to ORB function predictions.

The data and simulation models were provided by Trey McNeely, Ann Lee, Kimberly Wood, Nic Dalmasso and Pavel Khokhlov ([arXiv:1911.11089v3](https://arxiv.org/abs/1911.11089), [arXiv:2010.05783v3](https://arxiv.org/abs/1911.11089)).


## `orbconformal` package

[![Python](https://img.shields.io/badge/python-3.7-blue)]()
[![test & coverage](https://github.com/benjaminleroy/orb-tc-conformal/actions/workflows/code-check-and-coverage.yaml/badge.svg)](https://github.com/benjaminleroy/orb-tc-conformal/actions/workflows/code-check-and-coverage.yaml)
[![codecov](https://codecov.io/gh/benjaminleroy/orb-tc-conformal/branch/main/graph/badge.svg)](https://codecov.io/gh/benjaminleroy/orb-tc-conformal)
[![CodeFactor](https://www.codefactor.io/repository/github/benjaminleroy/orb-tc-conformal/badge?s=59656066a0a4a17814dd3d5f29b154e40fcc585e)](https://www.codefactor.io/repository/github/benjaminleroy/orb-tc-conformal)

Head to the [package](https://github.com/benjaminleroy/orb-tc-conformal/tree/main/package/orbconformal) part of this repo.


## useful development info:

### Compile `.md` files locally:

using grip (`pip install grib`)

```bash
grip -b file_name.md
```

### Developing with [`py-pkgs`](https://py-pkgs.org/)

#### using `poetry`

- package adding:
  dev only:
  ```bash
  poetry add --dev package
  ```
  for package:
  ```bash
  poetry add package
  ```
-  test running:
  ```bash
  poetry run pytest
  ```

- building the docs:
  in the `docs/` folder (using `numpy` documentation style):
  ```bash
  poetry run make html
  ```

- install package locally
  ```bash
  poetry install
  ```

  Note that sometimes deleting the .lock file can be helpful if there are major errors installing packages
