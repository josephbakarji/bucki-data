# Overview
 This code contains various machine learning methods for discovering dimensionless groups from simulation/experimental data. Check out the details in the Arxiv paper: [Dimensionally Consistent Learning with Buckingham Pi](https://arxiv.org/abs/2202.04643). Here's the abstract:

## Overview
In the absence of governing equations, dimensional analysis is a robust technique for extracting insights and finding symmetries in physical systems. Given measurement variables and parameters, the Buckingham Pi theorem provides a procedure for finding a set of dimensionless groups that spans the solution space, although this set is not unique. We propose an automated approach using the symmetric and self-similar structure of available measurement data to discover the dimensionless groups that best collapse this data to a lower dimensional space according to an optimal fit. We develop three data-driven techniques that use the Buckingham Pi theorem as a constraint: (i) a constrained optimization problem with a non-parametric input-output fitting function, (ii) a deep learning algorithm (BuckiNet) that projects the input parameter space to a lower dimension in the first layer, and (iii) a technique based on sparse identification of nonlinear dynamics (SINDy) to discover dimensionless equations whose coefficients parameterize the dynamics. We explore the accuracy, robustness and computational complexity of these methods as applied to three example problems: a bead on a rotating hoop, a laminar boundary layer, and Rayleigh-Bénard convection.

# Content summary
- `testcases`: tests all the methods on various problems: Blasius boundary layer, bead on a rotating hoop, Rayleigh-Benard problem. Showing basic usage and examples.
- `src`: contains the main machine learning code for methods: Constrained optimization, BuckiNet and other helper functions.
- `solvers`: contains numerical solvers to generate simulation data
- `data`: contains data that is simulated outside this code, i.e. the Rayleigh-Benard problem (the full simulation data is too large to include in this repo and is available upon request).

# Requirements

Tested with the following:
- Software dependencies: numpy 1.21, scipy 1.6, matplotlib 3.1, tensorflow 2.8, IPython 7.13, sklearn 0.24.1, sympy 1.8.
- OS: Unix (should work on windows but hasn't been tested).
- Hardware: GPUs are use if available but are not required. 
