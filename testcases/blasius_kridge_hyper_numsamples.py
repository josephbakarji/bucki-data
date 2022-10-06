import numpy as np
import numpy.random as rng
from scipy.integrate import odeint
import json

import os
import sys
sys.path.append('../solvers/')
sys.path.append('../src/')

from blasius import solve_blasius
from learning import KRidgeReg
from helper_functions import prettify_results, get_hyperparameter_list
import pdb


eta_inf=10
d_eta=0.01
nu = 1e-6       # Viscosity of water near room temperature  (m^2/s)
U_inf = 0.01    # m/s
xlim = [1e-3, 1e-1]

x, y, u, v = solve_blasius(xlim, nu, U_inf, eta_inf, d_eta)

yy, xx = np.meshgrid(y, x)
eta = yy*np.sqrt(U_inf/(xx*nu))  # Exact value of eta

Re = (U_inf/nu)*x
delta = 1.72 * np.sqrt(x*nu/U_inf)


dim_matrix = np.array([[1, 0], [1, 0], [1, -1], [2,  -1]]).T
names = ['x', 'y', 'U_{inf}', '\\nu']

# True vectors are in the nullspace
eta_vec = np.array([-0.5, 1, 0.5, -0.5])
Re_vec = np.array([1, 0, 1, -1])

# Inputs: [x, y, U, nu]
p = np.vstack([xx.flatten(),
               yy.flatten(),
               np.full(u.shape, U_inf).flatten(),
               np.full(u.shape, nu).flatten()]).T

# Outputs: [u/Uinf, v/U_inf]
q = np.vstack([u.flatten()/U_inf, v.flatten()/U_inf])

hyperparams = {
'nsamp': [20, 30, 60, 160, 275, 450],
'l1_reg' : [1e-3],
'gamma' : [1],
'kernel' : ['rbf'],
'alpha': [1e-4],
'use_test_set' : [True],
'normalize': [False],
'num_trials': [100],
'num_nondim': [1]
}

savedir = './'
savefile = 'blasius_kridge_hyper_numsamples.json'
data_list = []
hyperparams_list = get_hyperparameter_list(hyperparams)
for hyperp in hyperparams_list:
    params = {}
    for key, val in hyperp.items():
        params[key] = val

    print(params)

    # Randomly subsample observable and parameters
    nsamp = params['nsamp'] 
    l1_reg = params['l1_reg'] 
    alpha = params['alpha'] 
    kernel = params['kernel'] 
    gamma = params['gamma'] 
    use_test_set = params['use_test_set'] 
    normalize = params['normalize'] 
    num_trials = params['num_trials'] 
    frac_tol = 0.1
    max_denominator = 10
    num_nondim = params['num_nondim']

    idx = np.random.choice(np.arange(q.shape[1]), nsamp)
    outputs = q[0, idx].T
    inputs = p[idx, :]

    try:
        # Ridge Regression
        K = KRidgeReg(inputs, outputs, dim_matrix, num_nondim=num_nondim, #normalize=normalize,
        l1_reg=l1_reg, alpha=alpha, kernel=kernel, gamma=gamma)

        x_opt, x_list, loss_list = K.multi_run(ntrials=num_trials)

        for idx, x in enumerate(x_list):
            if x is not None:
                if type(x) == np.ndarray:
                    x_list[idx] = x.T.tolist()[0]


        data = {'x_list': x_list, 'loss_list':loss_list, 'params': params}

        if os.path.exists(savedir+savefile):
            f = open(savedir+savefile)
            data_list = json.load(f)
            f.close()
        
        data_list.append(data)

        with open(savefile, 'w') as g:
            json.dump(data_list, g)
    
    except ValueError or AttributeError:
        print('value error')
        pass
