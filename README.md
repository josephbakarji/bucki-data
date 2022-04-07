# Data-driven dimensional analysis
 This code contains various machine learning methods for data-driven non-dimensionalization based on simulation/experimental data. Check out the details in the Arxiv paper: [Dimensionally Consistent Learning with Buckingham Pi](https://arxiv.org/abs/2202.04643).

## Code summary:
- testcases: tests all the methods on various problems: Blasius boundary layer, bead on a rotating hoop, Rayleigh-Benard problem
- src: contains the main machine learning code for methods: Constrained optimization, BuckiNet and other helper functions.
- solvers: contains numerical solvers to generate simulation data
- data: contains data that is generated outside this code 
