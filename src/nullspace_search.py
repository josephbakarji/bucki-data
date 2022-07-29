import numpy as np
import numpy.random as rng
from scipy.optimize import minimize
import pdb
from math import isnan, isinf
from itertools import compress, product, combinations


# Very inefficient
def is_collinear(v1, v2):
    """
    Return True if v1 and v2 are collinear vectors
    """
    r = v1/v2
    x = []
    for i, ri in enumerate(r):
        if not isnan(ri) and not isinf(ri) and not isinf(-ri):
            x.append(ri)
    if not np.any(np.array(x)-x[0]):
        return True
    else:
        return False

def number_exists(num, nondim):
    """
    Return True if num (or a collinear to it) already exists in list 
    """
    if not np.any(num):
        return True
    for discovered in nondim:
        if is_collinear(num, discovered):
            return True
    return False

def get_nondim_numbers(dim_matrix, degree, num_params, num_nondim):
    """
    returns all possible dimensionless numbers in nullspace up to degree.
    """
    all_combinations = np.array(list(product(*[range(-degree, degree+1)] * num_params)))
    idxsort = np.argsort(np.sum(np.abs(all_combinations), axis=1))
    all_combinations = all_combinations[idxsort]
    nondim_list = []
    for i in range(all_combinations.shape[0]):
        pi_candidate = all_combinations[i, :]
        if not np.any(dim_matrix @ pi_candidate):
            if not number_exists(pi_candidate, nondim_list):
                nondim_list.append(pi_candidate)
    return nondim_list 

def fit_allnondim(fitting_class, nondim_list, num_nondim):
    """
    Fits all num_nondim combinations from nondim_list to fitting_class (function and data)
    Fitting class has to have a .loss method with input x (powers of parameters)
    """
    input_combinations = list(combinations(nondim_list, num_nondim))
    loss_list = []
    for x in input_combinations:
        # print(i/len(input_combinations))
        print(x)
        xarr = np.vstack(x)
        test_loss = fitting_class.loss(xarr)
        loss_list.append(test_loss)
        print('test loss = ', test_loss)
    return input_combinations, loss_list

