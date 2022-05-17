# Data-driven dimensional analysis
 This code contains various machine learning methods for data-driven non-dimensionalization based on simulation/experimental data. Check out the details in the Arxiv paper: [Dimensionally Consistent Learning with Buckingham Pi](https://arxiv.org/abs/2202.04643).

## Repository summary
- `testcases`: tests all the methods on various problems: Blasius boundary layer, bead on a rotating hoop, Rayleigh-Benard problem. Showing basic usage and examples.
- `src`: contains the main machine learning code for methods: Constrained optimization, BuckiNet and other helper functions.
- `solvers`: contains numerical solvers to generate simulation data
- `data`: originally contains data that is not simulated with this code (the rayleigh-benard simulation data is too large to include in this repo and is available upon request).


## Requirements

Tested with the following:
- Software dependencies: numpy 1.21, scipy 1.6, matplotlib 3.1, tensorflow 2.8, IPython 7.13, sklearn 0.24.1.
- OS: Unix (should work on windows but hasn't been tested).
- Hardware: GPUs are use if available but are not required. 
