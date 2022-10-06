import numpy as np
import pdb
import numpy.random as rng
from scipy.integrate import odeint

import sys
sys.path.append('../solvers/')
sys.path.append('../src/')

from blasius import solve_blasius
from learning import BuckyNet    
from helper_functions import prettify_results

eta_inf=15
d_eta=0.01
nu = 1e-6       # Viscosity of water near room temperature  (m^2/s)
U_inf = 0.01    # m/s
xlim = [1e-3, 1e-1]

x, y, u, v = solve_blasius(xlim, nu, U_inf, eta_inf, d_eta)

yy, xx = np.meshgrid(y, x)
eta = yy*np.sqrt(U_inf/(xx*nu))  # Exact value of eta

Re = (U_inf/nu)*x
delta = 1.72 * np.sqrt(x*nu/U_inf)


# # Optimization

dim_matrix = np.array([[1, 0], [1, 0], [1, -1], [2,  -1]]).T
names = ['x', 'y', 'U_{inf}', '\\nu']

# True vectors are in the nullspace
eta_vec = np.array([-0.5, 1, 0.5, -0.5])
Re_vec = np.array([1, 0, 1, -1])
# print('$\eta$ and $Re$ are in null-space:')
# print(dim_matrix @ eta_vec)
# print(dim_matrix @ Re_vec)

# Inputs: [x, y, U, nu]
p = np.vstack([xx.flatten(),
               yy.flatten(),
               np.full(u.shape, U_inf).flatten(),
               np.full(u.shape, nu).flatten()]).T

# Outputs: [u/Uinf, v/U_inf]
q = np.vstack([u.flatten()/U_inf, v.flatten()/U_inf])

# Randomly subsample observable and parameters
nsamp = 2000
idx = np.random.choice(np.arange(q.shape[1]), nsamp)
outputs = q[0, idx].T
inputs = p[idx, :]


## TODO: Set up hyperparameter search.

verbose = 1
num_layers = 3
num_nondim = 1
num_neurons = 40
activation = 'elu'
initializer = 'he_normal'
nepoch = 50000
patience = 600 
test_size = 0.2
nullspace_loss = 0.5 # Set to weight \in [0, 1] if want to turn on
l1_reg = 0.000
l2_reg = 0.000
adamlr = 0.002

B = BuckyNet(inputs, outputs, dim_matrix, num_nondim=num_nondim, num_layers=num_layers, num_neurons=num_neurons, activation=activation, verbose=verbose, initializer=initializer, nepoch=nepoch, 
     patience=patience, test_size=test_size, nullspace_loss=nullspace_loss, l1_reg=l1_reg, l2_reg=l2_reg, adamlr=adamlr)

x = B.single_run()

pdb.set_trace()

print(dim_matrix@x)
print(x)
for i in range(x.shape[1]):
    print(x[:, i]/x[1, i])
