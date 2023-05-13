import numpy as np
from typing import Callable
from scipy import linalg as linalg

from . import givens
from ._check import check_parameters


def minres(A, b, /, tol: float=1e-6, maxit: int=None, pre=None, x0=None):
    check_parameters(pre)

    n = b.size
    
    if not maxit:
        maxit = n
    
    if not pre:
        pre = lambda x: x
    
    if not x0:
        x0 = np.zeros(n)
    x = x0
    
    if not callable(A):
        A_Mut = lambda x: A.dot(x)
    else:
        A_Mut = A
    
    r = b - A_Mut(x0)
    z1 = pre(r)
    xi1 = np.sqrt(np.dot(r, z1))
    v1, z1 = r / xi1, z1 / xi1

    v0 = p0 = p1 = 0
    beta1 = s0 = s1 = 0
    c0 = c1 = 1

    flag, resvec = 1, np.array([xi1])
    for j in range(maxit):
        v = A_Mut(z1) - beta1 * v0
        alpha = np.dot(z1, v)
        v = v - alpha * v1

        y = pre(v)
        beta2 = np.sqrt(np.dot(v, y))

        eta, beta1 = -s0 * beta1, c0 * beta1

        delta = c1 * beta1 - s1 * alpha
        alpha = s1 * beta1 + c1 * alpha

        c0, s0 = c1, s1
        c1, s1, gamma = givens(alpha, beta2)
        xi, xi1 = c1 * xi1, s1 * xi1

        p = (z1 - delta * p1 - eta * p0) / gamma
        x = x + xi * p

        resvec = np.append(resvec, np.abs(xi1))
        if resvec[-1] < tol:
            flag = 0
            break
        
        beta1 = beta2
        p0, p1 = p1, p
        v0, v1, z1 = v1, v / beta2, y / beta2

    return x, {'exitflag': flag, 'iters': j + 1, 'resvec': resvec}