import numpy as np
from scipy import linalg


def s3mr(S, b, alpha, /, tol: float=1e-6, maxit: int=None, x0=None) -> tuple[np.array, dict]:
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
    c0, s_1, s0 = 1, 0, 0
    gamma1 = norm_r
    phi_tilde = gamma1
    w0, w1 = 0, r / gamma1
    p_1 = p0 = 0

    flag, resvec = 1, np.array([norm_r])
    for j in range(maxit):
        if resvec[-1] / norm_r < tol:
            flag = 0
            break

        w = S_Mut(w1) + gamma1 * w0
        gamma2 = linalg.norm(w)
        delta = np.sqrt(delta_tilde ** 2 + gamma2 ** 2)
        c1, s1 = delta_tilde / delta, gamma2 / delta
        delta_tilde = alpha * c1 + gamma2 * c0 * s1
        phi, phi_tilde = c1 * phi_tilde, -s1 * phi_tilde

        if j < 2:
            p = w1 / delta
        else:
            p = (w1 + gamma1 * s_1 * p_1) / delta
        
        x = x + phi * p
        resvec = np.append(resvec, np.abs(phi_tilde))

        gamma1 = gamma2
        w0, w1 = w1, w / gamma1
        p_1, p0 = p0, p_1
        c0, s_1, s0 = c1, s0, s1

    return x, {'exitflag': flag, 'iters': j + 1, 'resvec': resvec}