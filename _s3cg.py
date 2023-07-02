import numpy as np
from scipy import linalg


def s3cg(A, b, /, tol: float=1e-6, maxit: int=None, x0=None) -> tuple[np.array, dict]:
    n = b.size

    if not maxit:
        maxit = n
    
    if not x0:
        x0 = np.zeros(n)
    x = x0
    
    if not callable(A):
        A_Mut = lambda x: A.dot(x)
    else:
        A_Mut = A
    
    r = b - A_Mut(x0)
    p = r
    norm_r = linalg.norm(r)
    delta1 = norm_r ** 2

    flag, resvec = 1, np.array([norm_r])
    for j in range(maxit):
        if resvec[-1] / norm_r < tol:
            flag = 0
            break
            
        h = A_Mut(p)
        alpha = delta1 / np.dot(p, h)
        x = x + alpha * p
        r = r - alpha * h

        tmp = linalg.norm(r)
        delta2 = tmp ** 2
        resvec = np.append(resvec, tmp)

        beta = -delta2 / delta1
        p = r + beta * p

        delta1 = delta2

    return x, {'exitflag': flag, 'iters': j + 1, 'resvec': resvec}