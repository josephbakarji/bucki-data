import numpy as np
import numpy.random as rng
from scipy.integrate import odeint
import json

import os
import sys
sys.path.append('../solvers/')
sys.path.append('../src/')

from blasius import solve_blasius
from learning import BuckyNet
from helper_functions import prettify_results, get_hyperparameter_list, save_results
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

verbose = 0
activation = 'elu'
initializer = 'he_normal'
nepoch = 50000
patience = 600
test_size = 0.2
adamlr = 0.002

hyperparams = {
'nsamp': [10, 15, 20, 25, 30, 35, 50, 60, 65, 75, 100, 120, 150, 160, 275, 300, 450, 500, 700, 900, 1100, 1300, 1500, 1800, 2000],
'l1_reg' : [0.0],
'l2_reg' : [0.0],
'num_nondim': [1],
'nullspace_loss' : [0.5],
'num_layers' : [3],
'num_neurons' : [40]
}

savedir = './results/'
savefile = 'blasius_buckinet_hyper_numsamples.json'
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
    num_nondim = params['num_nondim']
    num_layers = params['num_layers']
    num_neurons =params['num_neurons'] 
    nullspace_loss = params['nullspace_loss'] 
    l1_reg = params['l1_reg'] 
    l2_reg = params['l2_reg'] 

    idx = np.random.choice(np.arange(q.shape[1]), nsamp)
    outputs = q[0, idx].T
    inputs = p[idx, :]

    try:
        B = BuckyNet(inputs, outputs, dim_matrix, num_nondim=num_nondim, num_layers=num_layers, num_neurons=num_neurons, activation=activation, verbose=verbose, initializer=initializer, nepoch=nepoch, 
            patience=patience, test_size=test_size, nullspace_loss=nullspace_loss, l1_reg=l1_reg, l2_reg=l2_reg, adamlr=adamlr)

        x = B.single_run()
        loss = B.model.evaluate(B.inputs_test, B.outputs_test)

        data = {'x': x.tolist(), 'loss': loss, 'params': params}

        save_results(data, savefile, savedir)
    
    except ValueError or AttributeError:
        print('value error')
        pass
