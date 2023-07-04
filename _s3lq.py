import numpy as np
from scipy import linalg


def s3lq(S, b, alpha, /, tol: float=1e-6, maxit: int=None, x0=None) -> tuple[np.array, dict]:
    n = b.shape[0]

    if not maxit:
        maxit = n
    
    if not x0:
        x0 = np.zeros(b.shape)
    x = x0
    
    if not callable(S):
        if len(b.shape) == 1:
            S_Mut = lambda x: S.dot(x)
        else:
            S_Mut = lambda x: S @ x
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
            flag = 0
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


def bls3lq(S, B, alpha, /, tol: float=1e-6, maxit: int=None, X0=None) -> tuple[np.ndarray, dict]:
    n, p = B.shape

    if not maxit:
        maxit = n
    
    if not X0:
        X0 = np.zeros((n, p))
    X = X0
    
    if not callable(S):
        S_Mut = lambda x: S @ x
    else:
        S_Mut = S
    
    R0 = B - (S_Mut(X0) + alpha * X0)
    W1, R = linalg.qr(R0, mode='economic')
    norm_r = linalg.norm(R)
    W0, P_tilde = 0, W1
    Xi_1 = Xi0 = 0
    H1 = R

    C_1, S_1, C0, S0 = [np.zeros((p, p)) for _ in range(4)]
    Zeros, AlphaI = np.zeros((p, p)), alpha * np.eye(p)

    flag, resvec = 1, np.array([norm_r])
    for j in range(maxit):
        if resvec[-1] / norm_r < tol:
            flag = 0
            break

        if j == 0:
            W = S_Mut(W1)
        else:
            W = S_Mut(W1) + W0 @ H1.T
        W2, H2 = linalg.qr(W, mode='economic')

        if j > 1:
            T = np.hstack( (Zeros, H1) )
            _blrotation_of_s3lq(T, C_1, S_1)
            
            Phi, Theta = T[:, :p], T[:, p:]
        
        if j == 1:
            Theta = H1

        if j > 0:
            T = np.hstack( (Theta, AlphaI) )
            _blrotation_of_s3lq(T, C0, S0)
            
            Omga = T[:, p:]
        
        if j == 0:
            Omga = AlphaI

        C_1, S_1 = C0.copy(), S0.copy()

        T = np.hstack( (Omga, -H2.T) )
        T1 = np.hstack( (P_tilde, W2) )
        _blrotation_of_s3lq(T, C0, S0, T1)
        
        Omega, P, P_tilde = T[:, :p], T1[:, :p], T1[:, p:]

        if j == 0:
            Xi = linalg.solve_triangular(Omega, R, lower=True)
        elif j == 1:
            Xi = Zeros
        else:
            Xi = -linalg.solve_triangular(Omega, Phi @ Xi_1, lower=True)
        
        Tmp = P @ Xi
        X = X + Tmp
        R0 = R0 - (S_Mut(Tmp) + alpha * Tmp)
        resvec = np.append(resvec, linalg.norm(R0))

        H1 = H2
        W0, W1 = W1, W2
        Xi_1, Xi0 = Xi0, Xi

    return X, {'exitflag': flag, 'iters': j + 1, 'resvec': resvec}


def _blrotation_of_s3lq(A, C, S, /, B=None):
    p = C.shape[0]
    for i in range(p):
        for k in range(p):
            index = p+i-k-1

            if B is not None:
                delta_tilde, gamma = A[i, index:index+2]
                delta = np.sqrt( delta_tilde ** 2 + gamma ** 2 )
                c, s = delta_tilde / delta, gamma / delta
                C[k, i], S[k, i] = c, s
            else:
                c, s = C[k, i], S[k, i]
            
            Rot_Mat = np.array([[c, -s], [s, c]])
            A[:, index:index+2] = A[:, index:index+2] @ Rot_Mat
            if B is not None:
                B[:, index:index+2] = B[:, index:index+2] @ Rot_Mat