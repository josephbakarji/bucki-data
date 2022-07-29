# Matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('axes', labelsize=20)
mpl.rc('axes', titlesize=20)
mpl.rc('figure', figsize=(6, 4))
# %config InlineBackend.figure_format = 'retina'

from numpy.linalg import matrix_rank
import sys
sys.path.append('../src')
sys.path.append('../solvers')
from rotating_hoop import RotatingHoop
from learning import BuckyNet
# from nullspace_search import get_nondim_numbers, fit_allnondim
from helper_functions import prettify_results, save_results, get_hyperparameter_list    




nsamples = int(5e3)
output_type = 'svd' # options: 'dynamic', 'static', 'svd' - 
# num_nondim = 3 # !! Recomputed below - num_nondim=3 also works for 'svd' in some cases !!
# num_modes = 5
tsteps = 100 # For 'dynamic', use smaller number of tsteps - dynamic still takes too long
tend = 2
phi_init = [1, 0]

verbose = 0
# num_layers = 3
# num_neurons = 50
# nullspace_loss = .8 # Set to weight \in [0, 1] if want to turn on
# l1_reg = 0.000
activation = 'elu'
initializer = 'he_normal'
nepoch = 1000
patience = 100
test_size = 0.15
l2_reg = 0.000
adamlr = 0.002

hyperparams = {
'num_modes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
'num_nondim': [2, 3],
'l1_reg': [0.0, 1e-6, 1e-3],
'num_layers': [2, 3],
'num_neurons': [20, 40, 60],
'nullspace_loss': [0.5, 0.8, 1.0]
}

savedir = './'
savefile = 'hoop_buckinet_hyper_modes.json'
data_list = []
hyperparams_list = get_hyperparameter_list(hyperparams)

for hyperp in hyperparams_list:
    params = {}
    for key, val in hyperp.items():
        params[key] = val

    print(params)

    num_nondim = params['num_nondim']
    num_modes = params['num_modes']
    l1_reg = params['l1_reg']
    num_layers = params['num_layers'] 
    num_neurons = params['num_neurons'] 
    nullspace_loss = params['nullspace_loss'] 

    ## Get solution
    R = RotatingHoop(nsamples=nsamples, output_type=output_type, modes=num_modes, time_steps=tsteps, tend=tend, phi0=phi_init)
    inputs, outputs = R.get_data()
    dim_matrix = R.get_dim_matrix()

    B = BuckyNet(inputs, outputs, dim_matrix, num_nondim=num_nondim, num_layers=num_layers, num_neurons=num_neurons, activation=activation, verbose=verbose, initializer=initializer, nepoch=nepoch, 
                patience=patience, test_size=test_size, nullspace_loss=nullspace_loss, l1_reg=l1_reg, l2_reg=l2_reg, adamlr=adamlr)


    x = B.single_run()

    data = {'params':params, 'x': x.tolist()}

    save_results(data, savefile, savedir)


    idxs = [1, 1, -1]
    x1_norm = x[:, 0]/x[idxs[0], 0]
    x2_norm = x[:, 1]/x[idxs[1], 1]
    # x3_norm = x[:, 2]/x[idxs[2], 2]
    # Pi, names = R.get_dim_matrix(include_names=True)

    print(x)
    print('-------------')
    print(x1_norm)
    print(x2_norm)
    # print(x3_norm)
    print('-------------')
    # print(x@Pi)
