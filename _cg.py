import numpy as np
from typing import Callable
from scipy import linalg
from ._check import check_parameters


def cg(A, b, /, tol: float=1e-6, maxit: int=None, pre=None, x0=None):
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

    z = pre(r)
    p = z
    delta1 = np.dot(z, r)

    flag, resvec = 1, np.array([linalg.norm(r)])
    for j in range(maxit):
        if resvec[-1] < tol:
            flag = 0
            break

        h = A.dot(p)
        alpha = delta1 / np.dot(p, h)
        x = x + alpha * p
        r = r - alpha * h

        resvec = np.append(resvec, linalg.norm(r))

        z = pre(r)
        delta2 = np.dot(z, r)
        beta = delta2 / delta1
        p = z + beta * p

        delta1 = delta2

    return x, {'exitflag': flag, 'iters': j + 1, 'resvec': resvec}