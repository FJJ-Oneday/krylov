import numpy as np
from scipy import linalg


def s3lq(S, b, alpha, /, tol: float=1e-6, maxit: int=None, x0=None) -> tuple[np.array, dict]:
    n = b.size

    if not maxit:
        maxit = n
    
    if not x0:
        x0 = np.zeros(n)
    x = x0
    
    if not callable(S):
        S_Mut = lambda x: S.dot(x)
    else:
        S_Mut = S
    
    r = b - (S_Mut(x0) + alpha * x0)


    norm_r = linalg.norm(r)
    delta_tilde = alpha
    c0, s0, s_1, gamma1 = 1, 0, 0, norm_r
    xi_1 = xi0 = 0
    w0, w1 = 0, r / gamma1
    p_tilde = w1

    flag, resvec = 1, np.array([norm_r])
    for j in range(maxit):
        if resvec[-1] / norm_r < tol:
            flag = 1
            break

        w = S_Mut(w1) + gamma1 * w0
        gamma2 = linalg.norm(w)
        delta = np.sqrt(delta_tilde ** 2 + gamma2 ** 2)
        c1, s1 = delta_tilde / delta, -gamma2 / delta
        delta_tilde = alpha * c1 - gamma2 * c0 * s1

        if j == 0:
            xi1 = gamma1 / delta
        else:
            xi1 = -gamma1 * s_1 * xi_1 / delta
        
        w0, w1 = w1, w / gamma2

        p = c1 * p_tilde + s1 * w1
        x = x + xi1 * p
        p_tilde = c1 * w1 - s1 * p_tilde

        r = r - xi1 * (S_Mut(p) + alpha * p)
        resvec = np.append(resvec, linalg.norm(r))

        gamma1 = gamma2
        c0, s_1, s0 = c1, s0, s1
        xi_1, xi0 = xi0, xi1

    return x, {'exitflag': flag, 'iters': j + 1, 'resvec': resvec}