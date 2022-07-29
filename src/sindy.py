import numpy as np
from itertools import product, combinations
from scipy import linalg

import sympy as sym

# Simple sequentially thresholded least squares optimizer (could also use anything from pySINDy)
class STLSQ():
    def __init__(self, threshold=0.0, cond=None, max_iter=10, normalize=False):
        self.threshold = threshold
        self.cond = cond
        self.max_iter = max_iter
        self.normalize = normalize
        
    def fit(self, lib, dy, row_weights=None):
        
        # Row weights control relative weighting of different parameters
        if row_weights is None:
            row_weights = np.ones(lib.shape[0])
            
        # Column weights to normalize features (candidate functions)
        col_weights = np.ones(lib.shape[1])
        if self.normalize:
            col_weights = 1 / np.max(abs(lib), axis=0)
            
        lib_w = ( lib * col_weights[None, :] ) * row_weights[:, None]
        lhs_w = dy * row_weights[:, None]
        
        # Initial least-squares guess
        Xi = linalg.lstsq(lib_w, lhs_w, cond=self.cond)[0] * col_weights[:, None]
        
        # Iterate and threshold
        for k in range(self.max_iter):
            for i in range(Xi.shape[1]):
                small_idx = (abs(Xi[:, i]) < self.threshold)  # Find small indices
                Xi[small_idx, i] = 0   # Threshold
                big_idx = np.logical_not( small_idx[:] )
                # Regress dynamics onto remaining terms to find sparse Xi
                Xi[big_idx, i]  = linalg.lstsq(lib_w[:, big_idx], lhs_w[:, i], cond=self.cond)[0] * col_weights[big_idx]
        self.coef_ = Xi.T


class Quantity():
    def __init__(self, name, dim, sym_name=None):
        self.name = name   # String name
        self.dim = dim     # Array of dimensions, e.g. [2, 1, 0] = [length, width, time]
        if sym_name is None: sym_name = name
        self.symbol = sym.symbols([sym_name]).pop()  # Sympy symbol of the parameter
        
class System():
    """
    vectorize - function expecting a dict 'expt' and list of parameters 'Pi'
                that should return inputs (y), outputs (dy), params (p), and weights (w)
    """
    def __init__(self, qty_list, param_list):
        self.qty_list = qty_list
        self.param_list = param_list
        self.expts = []
        
    def param_names(self):
        return [p.name for p in self.param_list]
    
    def param_symbols(self):
        return [p.symbol for p in self.param_list]
    
    def param_vec(self, expt):
        return np.array([expt[key] for key in self.param_names()])
    
    def qty_names(self):
        return [q.name for q in self.qty_list]
        
    def add_expt(self, q, p):
        # Check that q and p contain all the quantities and parameters for the system,
        #   and vice versa
        assert sorted(q.keys()) == sorted(self.qty_names()) and \
                sorted(p.keys()) == sorted(self.param_names())
        
        # Add a new dictionary representing this experiment
        self.expts.append({**q, **p})
        
    def get_dim_matrix(self):
        return np.vstack([p.dim for p in self.param_list]).T
    
    def get_param_matrix(self):
        """
        Return the matrix P: [n_expts x n_params] where each row has parameters for one experiment
        """
        return np.vstack( [self.param_vec(expt) for expt in self.expts] )
    
class ParametricLibrary():
    def __init__(self, q_fns, p_fns):
        self.q_fns = q_fns
        self.p_fns = p_fns
        
    def __call__(self, Q, pi):
        return np.vstack(
            [ [ f(Q)*g(pi) for f in self.q_fns ] for g in self.p_fns ]
        ).T
    
    
def normalized_l2_loss(dq, Theta, Xi):
    return linalg.norm(dq.T - Theta @ Xi) / linalg.norm(dq.T)
    
def l2_loss(dq, Theta, Xi):
    return linalg.norm(dq.T - Theta @ Xi)
    
class DimensionlessSINDy():
    def __init__(self,
                 sys,
                 lib,
                 vectorize,
                 opt=STLSQ(),
                 nondim_degree=2):
        self.sys = sys
        self.lib = lib
        self.opt = opt
        
        # Generate vectors in nullspace (or time-dimensions) of the dimensional matrix
        self.dim_list = self.get_nondim_numbers(degree=nondim_degree)
        self.vectorize_expt = vectorize
        
    def vectorize_all(self, dim_vecs):
        """
        For a given set of candidate scalings, stack all the data
        Inputs:
            dim_vecs - list of exponent vectors (one for each dimensionless parameter)
        
        Outputs:
            q - the quantity of interest
            dq - dimensionless derivative (or other target quantity)
            p - the dimensionless parameters used in the library
            w - the relative weights of different experiments
        """
        # All dimensionless parameters (n_expt x n_params)
        P = self.sys.get_param_matrix()
        Phi = np.array(dim_vecs).T
        Pi = np.exp( np.log(P) @ Phi )
            
        expt_vecs = [self.vectorize_expt(expt, Pi[idx, :]) for (idx, expt) in enumerate(self.sys.expts)]
        q, dq, p, w = [np.hstack([vecs[i] for vecs in expt_vecs]) for i in range(len(expt_vecs[0]))]
        return q, dq, p, w
            
    def get_nondim_numbers(self, degree=2):
        """
        By default, returns all possible dimensionless numbers in nullspace up to degree.

        Can modify to search for vectors of any dimension (e.g. time) using dim_vecs

        If dim_vecs is included, should give the dimensions of time in dim_matrix rows
            e.g. [0, 0, 1]
            In this case, also returns all possible timescales
        """
        dim_matrix = self.sys.get_dim_matrix()
        dim_vecs = [q.dim for q in self.sys.qty_list]
        
        all_combinations = np.array(list(product(*[range(-degree, degree+1)] * dim_matrix.shape[1])))
        idxsort = np.argsort(np.sum(np.abs(all_combinations), axis=1))
        all_combinations = all_combinations[idxsort]
        vec_list = [[] for i in range(len(dim_vecs))]
        timescale_list = []
        for i in range(all_combinations.shape[0]):
            phi_candidate = all_combinations[i, :]
            if np.any(phi_candidate):
                # Check if candidate has correct dimensions for each input vector
                for (idx, dim_vec) in enumerate(dim_vecs):
                    if not np.any(dim_matrix @ phi_candidate - dim_vec):
                        vec_list[idx].append(phi_candidate)
        if len(dim_vecs)==1:
            return vec_list[0]
        else:
            return vec_list
        
    def fit_all_dynamics(self, loss=normalized_l2_loss, num_nondim=1, verbose=False):
        if verbose:
            from IPython.display import display, Math

        timescale_list, nondim_list = self.dim_list
        p_sym = self.sys.param_symbols()
        
        input_combinations = list(combinations(nondim_list, num_nondim))

        param_names = []
        param_list = []
        loss_list = []
        for T_vec in timescale_list:
            for Q_vecs in input_combinations:
                q, dq, pi, w = self.vectorize_all((T_vec, *Q_vecs))

                # Fit SINDy model
                Theta = self.lib(q, pi)
                self.opt.fit(Theta, dq.T, row_weights=w)
                Xi = self.opt.coef_.T

                # Save results
                loss_list.append(loss(dq, Theta, Xi))

                # Save this parameter combination
                test_params = [sym.prod(p_sym**T_vec),
                               *[sym.prod(p_sym**vec) for vec in Q_vecs]]
                param_names.append(test_params)
                param_list.append( (T_vec, *Q_vecs) )
                if verbose:
                    display(Math(sym.latex(test_params)))
                    print('test loss = ', test_loss)
                    print('----------------------------')
        return param_list, param_names, np.array(loss_list)