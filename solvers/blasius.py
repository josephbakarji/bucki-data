import numpy as np
import pdb
import numpy.random as rng

from scipy.integrate import odeint
from scipy.optimize import minimize, root
import matplotlib.pyplot as plt
import matplotlib as mpl

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

"""

For constant free stream flow, the [Blasius similarity solution](https://en.wikipedia.org/wiki/Blasius_boundary_layer#Blasius_equation_-_First-order_boundary_layer) gives the boundary layer behavior a s a function of the similarity variable $$\eta = \frac{y}{\delta(x)} = y \sqrt{\frac{U_\infty}{ \nu x}},$$
where $\delta(x)$ is the boundary layer thickness. Under Blasius' assumptions the flow is determined by the solution to the nonlinear boundary value problem
$$ f'''(\eta) + \frac{1}{2}f''(\eta) f(\eta), $$
$$ f(0) = f'(0) = 0, f'(\infty) = 1$$

Since this is a third-order equation, if a value for $f''(0)$ can be found which satisfies the infinite boundary condition, the self-similarity function $f(\eta)$ can be found numerically (e.g. with Runge-Kutta integration). This unknown "initial condition" can be established through root-finding.

Once $f(\eta)$ is known, the boundary layer profile can be found via
$$ u(x, y) = U_\infty f'(\eta), \hspace{1cm} v(x, y) = \frac{U_\infty}{2}\sqrt{\frac{\nu}{ U_\infty x}} [\eta f'(\eta) - f(\eta)]. $$
In other words, $u/U_\infty$ depends only on $\eta$, but $v/U_\infty$ depends on $\eta$ and $Re_x$

Obviously the equation cannot be solved numerically at infinity, but since it quickly approaches a linear solution, solving the problem on $\eta \in (0, 10)$ should be sufficient.

To solve, we need to define variables to reduce the third-order ODE to a first-order system of ODEs. This is done by defining
$$ f(\eta) = f_0 $$
$$ f'(\eta) = f_1 $$
$$ f''(\eta) = f_2 $$
Then the BVP becomes
$$f_0' = f_1 $$
$$ f_1' = f_2 $$
$$ f_2' = -\frac{1}{2}f_2 f_0 $$
with boundary conditions
$$f_0(0) = f_1(0) = 0$$
$$f_1(\infty) = 1$$

To solve with scipy, need to define a function to evaluate the ODE (for forward integration) and a function to evaluate the error in the boundary conditions:

"""


def blasius_rhs(f):
    """RHS of Blasius equation recast as first order nonlinear ODE
    f[0] = f
    f[1] = f'
    f[2] = f''
    """
    return np.array([f[1], f[2], -f[0]*f[2]/2])

def bc_fn(f0, eta):
    """Solve with unknown initial condition as guess and evaluate at upper boundary"""
    f = odeint(lambda f, t: blasius_rhs(f), f0, eta)
    # return discrepancy between upper boundary and desired f[2] = 1
    return [f0[0], f0[1], f[-1, 1] - 1]


def solve_blasius(xlim=[1e-3, 1e-1], nu=1e-6, U_inf=0.01, eta_inf=10, d_eta=0.01):

    # `bc_fn` should return all zeros when the boundary conditions are satisfied. 
    # Then we can use `scipy.optimize.root` to find the unknown value of $f''(0)$ and `scipy.integrate.odeint` 
    # to solve the equation. The results can be compared against the exact result from Boyd (1999):
    # John P. Boyd, "The Blasius function in the complex plane," *Experimental Mathematics*, 8 (1999), 381-394.

    # Solve root-finding problem for unknown initial condition
    F_init = [0, 0, 0] # Initial guess for unknown initial condition
    eta = np.arange(0, eta_inf, d_eta)
    opt_res = root(bc_fn, F_init, args=eta, tol=1e-4)
    F0 = [0, 0, opt_res.x[2]]

    # Evaluate with resulting initial conditions
    f = odeint(lambda y, t: blasius_rhs(y), F0, eta)

    x = np.linspace(xlim[0], xlim[1], 100)

    Re = (U_inf/nu)*x
    delta = 1.72 * np.sqrt(x*nu/U_inf)

    y = np.linspace(1e-4, 2*max(delta), 100)
    yy, xx = np.meshgrid(y, x)

    u = np.zeros([len(x), len(y)])
    v = np.zeros(u.shape)

    eta = yy*np.sqrt(U_inf/(xx*nu))  # Exact value of eta

    for i in range(len(x)):
        f = odeint(lambda y, t: blasius_rhs(y), F0, eta[i, :])
        u[i, :] = U_inf * f[:, 1]
        v[i, :] = (0.5*U_inf/np.sqrt(Re[i])) * (eta[i, :]*f[:, 1] - f[:, 0])

    return x, y, u, v





if __name__=="__main__":
    
    eta_inf=10
    d_eta=0.01
    nu = 1e-6       # Viscosity of water near room temperature  (m^2/s)
    U_inf = 0.01    # m/s
    xlim = [1e-3, 1e-1]

    x, y, u, v = solve_blasius(xlim, nu, U_inf, eta_inf, d_eta)

    Re = (U_inf/nu)*x
    delta = 1.72 * np.sqrt(x*nu/U_inf)

    plt.figure(figsize=(6, 2))
    plt.plot(x, delta)
    plt.grid()
    plt.xlabel('$x$')
    plt.ylabel(r'$\delta$')

    plt.figure(figsize=(6, 2))
    plt.pcolor(x, y, u.T, shading='auto', cmap='bone')
    plt.plot(x, delta, c='w')
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.figure(figsize=(4, 2))

    yy, xx = np.meshgrid(y, x)
    eta = yy*np.sqrt(U_inf/(xx*nu))  # Exact value of eta

    plt.scatter(u.flatten()/U_inf, eta.flatten(), c='k', s=0.1)
    plt.xlim()
    plt.grid()
    plt.xlabel(r'$u$')
    plt.ylabel(r'$\eta$')

plt.show()