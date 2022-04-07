import numpy as np
import numpy.random as rng

from scipy.integrate import odeint


class DuffingEqn:
    def __init__(self, nsamples=int(1e2), output_type='svd', modes=4, time_steps=500, tend=100, phi0=[0, 0]):
        self.nsamples = nsamples
        self.time_steps = time_steps
        self.output_type = output_type
        self.modes = modes
        self.tend = tend
        self.phi0 = phi0

    def get_data(self):
        # [d, a, b, g, w] and t
        ## Better way to find time_steps
        # period = 2*np.pi/omega
        # dt = 2*np.pi/omega / self.dt_per_period
        # tsteps = int(period / dt)

        d = np.random.uniform(1e-2, 1, self.nsamples)
        a = np.random.uniform(1e-2, 2, self.nsamples)
        b = np.random.uniform(1e-2, 3, self.nsamples)
        g = np.random.uniform(1e-2, 1, self.nsamples) 
        w = np.random.uniform(1e-2, 4, self.nsamples)
        t = np.linspace(0, self.tend, self.time_steps)
        p = np.vstack([d, a, b, g, w]).T
        
        if self.output_type == 'dynamic':
            # Using [1:] rid of the t=0 data point for which log(p) = inf
            phi = self.solve(d, a, b, g, w, t)
            p_t = np.concatenate((np.array([p[0, :]]*len(t[1:])), t[1:, None]), axis=1) 
            phi_t = phi[0, 1:].T
            for i in range(1, p.shape[0]):
                pext = np.concatenate((np.array([p[i, :]]*len(t[1:])), t[1:, None]), axis=1)
                p_t = np.concatenate((p_t, pext), axis=0)
                phi_t = np.concatenate((phi_t, phi[i, 1:].T), axis=0)

            return p_t, phi_t 
            
        elif self.output_type == 'svd':
            p = np.vstack([d, a, b, g, w]).T
            phi = self.solve(d, a, b, g, w, t)
            U, S, V = self.get_svd(phi)
            return p, V[:self.modes, :].T



    def solve(self, d, a, b, g, w, t):
        # Initial conditions: x, xdot
        phi = np.zeros([self.nsamples, len(t)])
        for i in range(self.nsamples):
            phi[i, :] = self.run_sim(d[i], a[i], b[i], g[i], w[i], self.phi0, t)[:, 0]
        return phi

    def run_sim(self, d, a, b, g, w, phi0, t):
        return odeint(lambda y, t: self.rhs(y, t, [d, a, b, g, w]), phi0, t)

    def rhs(self, X, t, p):
        d, a, b, g, w = p
        return np.array([X[1], -d*X[1] - a*X[0] - b*X[0]**3 + g * np.cos(w*t)])

    def get_svd(self, phi):
        U, S, V = np.linalg.svd(phi.T, full_matrices=False)
        return U, S, V

    def get_dim_matrix(self, include_names=False):
        if self.output_type == 'dynamic':
            Pi = np.array([[0, 0, -1],[0, 0, -2], [0, -2, -2], [0, 0, -2], [0, 0, -1], [0, 0, 1]]).T
            names = ['d', 'a', 'b', 'g', 'w', 't']
        else:
            Pi = np.array([[0, 0, -1],[0, 0, -2], [0, -2, -2], [0, 0, -2], [0, 0, -1]]).T
            names = ['d', 'a', 'b', 'g', 'w']
        if include_names:
            return Pi, names
        else:
            return Pi

    def get_true_nondim(self): # Not general enough. Fix.
        if self.output_type == 'dynamic':
            return np.array([[0, 1, 0, -1, 2, 0], [2, 1, -2, 1, 0, 0], [1, 0, -1, 1, 0, 1]])
        else:
            return np.array([[0, 1, 0, -1, 2], [2, 1, -2, 1, 0]])
