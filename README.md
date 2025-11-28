# dynode

[![PyPI version shields.io](https://img.shields.io/pypi/v/dynode.svg)](https://pypi.python.org/pypi/dynode/)
![](https://github.com/freol35241/dynode/workflows/dynode/badge.svg)
[![codecov](https://codecov.io/gh/freol35241/dynode/branch/master/graph/badge.svg)](https://codecov.io/gh/freol35241/dynode)
![docs](https://github.com/freol35241/dynode/workflows/docs/badge.svg)


Dynode solves equations of the form ```y' = f(t, y)``` using SciPy's [ode solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html) but allows ```f``` to be modelled in a modular, object-oriented fashion using the notions of separate ```Systems``` that expose ```states``` and their corresponding derivatives, ```ders```. ```f``` may then be composed of an arbitraily complex collection of, connected or unconnected, ```Systems```.

## Performance

Dynode supports optional [Numba](https://numba.pydata.org/) JIT compilation for accelerating compute-intensive systems. See the [documentation](https://freol35241.github.io/dynode/) for examples showing 2-20x speedups.

```bash
pip install numba  # Optional dependency
```

Documentation is available here: https://freol35241.github.io/dynode/

## License

Distributed under the terms of the MIT license, `dynode` is free and open source software


