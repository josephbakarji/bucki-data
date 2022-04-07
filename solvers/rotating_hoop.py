import numpy as np
import numpy.random as rng

from scipy.integrate import odeint


class RotatingHoop:
    def __init__(self, nsamples=int(1e2), output_type='svd', modes=4, time_steps=500, tend=100, phi0=[1, 0]):
        self.nsamples = nsamples
        self.output_type = output_type
        self.modes = modes
        self.time_steps = time_steps
        self.tend = tend
        self.phi0 = phi0
    

    def get_data(self):
        # [m, R, b, g, w] and t
        m = np.random.uniform(1e-2, 1, self.nsamples)
        R = np.random.uniform(1e-2, 2, self.nsamples)
        b = np.random.uniform(1, 10, self.nsamples)
        g = np.random.uniform(9, 11, self.nsamples) ## ??
        w = np.random.uniform(1e-2, 4, self.nsamples)
        t = np.linspace(0, self.tend, self.time_steps)
        p = np.vstack([m, R, b, g, w]).T
        
        if self.output_type == 'dynamic':
            # Using [1:] rid of the t=0 data point for which log(p) = inf
            phi = self.solve(m, R, b, g, w, t)
            p_t = np.concatenate((np.array([p[0, :]]*len(t[1:])), t[1:, None]), axis=1) 
            phi_t = phi[0, 1:].T
            for i in range(1, p.shape[0]):
                pext = np.concatenate((np.array([p[i, :]]*len(t[1:])), t[1:, None]), axis=1)
                p_t = np.concatenate((p_t, pext), axis=0)
                phi_t = np.concatenate((phi_t, phi[i, 1:].T), axis=0)
            

            return p_t, phi_t 
            
        elif self.output_type == 'svd':
            p = np.vstack([m, R, b, g, w]).T
            phi = self.solve(m, R, b, g, w, t)
            U, S, V = self.get_svd(phi)
            return p, V[:self.modes, :].T

        elif self.output_type == 'static':
            gam = R * w**2 / g
            phi = np.zeros_like(gam)
            phi[gam > 1] = np.arccos(1/gam[gam > 1])
            return p, phi


    def solve(self, m, R, b, g, w, t):
        eps = m**2 * g * R / b**2
        gam = R * w**2 / g
        phi = np.zeros([self.nsamples, len(t)])
        for i in range(self.nsamples):
            phi[i, :] = self.run_sim(eps[i], gam[i], self.phi0, t)[:, 0]
        return phi

    def run_sim(self, eps, gamma, phi0, t):
        return odeint(lambda y, t: self.rhs(y, [eps, gamma]), phi0, t)

    def rhs(self, y, p):
        e, g = p
        return np.array([y[1],(-y[1] - np.sin(y[0]) + g*np.sin(y[0])*np.cos(y[0])) / e])

    def get_svd(self, phi):
        U, S, V = np.linalg.svd(phi.T, full_matrices=False)
        return U, S, V

    def get_dim_matrix(self, include_names=False):
        if self.output_type == 'dynamic':
            Pi = np.array([[0, 1, 0], [1, 0, 0], [1, 1, -1], [1, 0, -2], [0, 0, -1], [0, 0, 1]]).T
            names = ['m', 'R', 'b', 'g', 'w', 't']
        else:
            Pi = np.array([[0, 1, 0], [1, 0, 0], [1, 1, -1], [1, 0, -2], [0, 0, -1]]).T
            names = ['m', 'R', 'b', 'g', 'w']
        if include_names:
            return Pi, names
        else:
            return Pi

    def get_true_nondim(self): # Not general enough. Fix.
        if self.output_type == 'dynamic':
            return np.array([[0, 1, 0, -1, 2, 0], [2, 1, -2, 1, 0, 0], [1, 0, -1, 1, 0, 1]])
        else:
            return np.array([[0, 1, 0, -1, 2], [2, 1, -2, 1, 0]])
