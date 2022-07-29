import numpy as np
import pdb
import numpy.random as rng
from scipy.integrate import odeint

import sys
sys.path.append('../solvers/')
sys.path.append('../src/')

from blasius import solve_blasius
from learning import KRidgeReg
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
nsamp = 400
idx = np.random.choice(np.arange(q.shape[1]), nsamp)
outputs = q[0, idx].T
inputs = p[idx, :]


## Consider hyperparameter optimization over l1_reg and alpha
l1_reg = 1e-3 ## Solution is sensitive to L1
alpha = 1e-4
kernel = 'rbf'
gamma = 1 
use_test_set = False
normalize = False 
num_trials = 30
frac_tol = 0.1
max_denominator = 10
num_nondim = 1

# Ridge Regression
K = KRidgeReg(inputs, outputs, dim_matrix, num_nondim=num_nondim, use_test_set=use_test_set,
 l1_reg=l1_reg, alpha=alpha, kernel=kernel, gamma=gamma)

x, x_list, loss_list = K.multi_run(ntrials=num_trials)

x_list_arr = []
x_list_arr_idx = []
xdiv = 0
for idx, x in enumerate(x_list):
    if x is not None:
        if type(x) != np.ndarray:
            xdiv += 1
        else:
            x_list_arr.append(x)    
            x_list_arr_idx.append(idx)

x_list_arr = x_list_arr
x_list_arr_idx = np.array(x_list_arr_idx)

num_none = len(x_list) - len(x_list_arr)

x_list_norm1 = []
x_list_norm2 = []
for x in x_list_arr:
    print(x)
    x_list_norm1.append( x[:, 0]/x[3, 0] )
    # x_list_norm2.append( x[:, 1]/x[3, 1] )

loss_list_arr = np.array(loss_list)
# losses = loss_list_arr[x_list_arr_idx]


print(num_none)
for i in range(len(x_list_norm1)):
    print(x_list_norm1[i])
    # print(x_list_norm2[i])
    # print(losses[i])

