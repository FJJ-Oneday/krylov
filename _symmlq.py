import numpy as np
from typing import Callable
from scipy import linalg as linalg

from . import givens
from ._check import check_parameters


def symmlq(A, b, /, tol: float=1e-6, maxit: int=None, 
           pre: Callable[[np.ndarray], np.ndarray]=None, x0=None) -> tuple[np.ndarray, dict]:
    check_parameters(pre)

    n = b.size

    if not callable(A):
        A_Mult = lambda x: A.dot(x)
    else:
        A_Mult = A

    if x0 is None:
        x0 = np.zeros(n)
    xl = x0
    r = b - A_Mult(xl)
    
    if maxit is None:
        maxit = n

    if not pre:
        pre = lambda x: x
    
    z1 = pre(r)
    eta = np.sqrt(np.dot(r, z1))
    v0, v1, z1 = 0, r / eta, z1 / eta

    c0 = c1 = 1
    xi0 = xi1 = s0 = s1 = beta1 = 0

    pbar = z1

    flag, resvec = 1, np.array([])
    for j in range(maxit):
        v = A_Mult(z1) - beta1 * v0
        alpha = np.dot(z1, v)
        v = v - alpha * v1

        y = pre(v)
        beta2 = np.sqrt(np.dot(v, y))

        epslion, t1 = -s0 * beta1, c0 * beta1
        delta, t2 = c1 * t1 - s1 * alpha, s1 * t1 + c1 * alpha

        c2, s2, gamma = givens(t2, beta2)

        if j == 0:
            xi2 = eta / gamma
        else:
            xi2 = -(epslion * xi0 + delta * xi1) / gamma

        # xc = xl + c2 * xi2 * pbar
        # rc = np.append(rc, np.abs(c1 * s2 / c2) * rc[-1])
        # if rc[-1] < tol:
        #     flag, x = 0, xc
        #     break

        resvec = np.append(resvec, np.sqrt( (gamma*xi2)**2 + (s1*beta2*xi1)**2 ))
        if resvec[-1] < tol:
            flag = 0
            break
        
        v2, z1 = v / beta2, y / beta2
        p = c2 * pbar - s2 * z1
        pbar = s2 * pbar + c2 * z1

        xl = xl + xi2 *  p

        v0, v1 = v1, v2
        beta1 = beta2
        c0, c1 = c1, c2
        s0, s1 = s1, s2
        xi0, xi1 = xi1, xi2

    return xl, {'exitflag': flag, 'iters': j + 1, 'resvec': resvec}