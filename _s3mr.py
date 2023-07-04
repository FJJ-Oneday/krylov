import numpy as np
from scipy import linalg


def s3mr(S, b, alpha, /, tol: float=1e-6, maxit: int=None, x0=None) -> tuple[np.array, dict]:
    n = b.shape[0]

    if not maxit:
        maxit = n
    
    if not x0:
        x0 = np.zeros(b.shape)
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
        p_1, p0 = p0, p
        c0, s_1, s0 = c1, s0, s1

    return x, {'exitflag': flag, 'iters': j + 1, 'resvec': resvec}


def gls3mr(S, B, alpha, /, tol: float=1e-6, maxit: int=None, X0=None) -> tuple[np.array, dict]:
    n = B.shape

    if not maxit:
        maxit = n
    
    if not X0:
        X0 = np.zeros(B.shape)
    X = X0
    
    if not callable(S):
        S_Mut = lambda x: S @ x
    else:
        S_Mut = S
    
    R = B - (S_Mut(X0) + alpha * X0)

    norm_r = linalg.norm(R)
    delta_tilde = alpha
    c0, s_1, s0 = 1, 0, 0
    gamma1 = norm_r
    phi_tilde = gamma1
    W0, W1 = 0, R / gamma1
    P_1 = P0 = 0

    flag, resvec = 1, np.array([norm_r])
    for j in range(maxit):
        if resvec[-1] / norm_r < tol:
            flag = 0
            break

        W = S_Mut(W1) + gamma1 * W0
        gamma2 = linalg.norm(W)
        delta = np.sqrt(delta_tilde ** 2 + gamma2 ** 2)
        c1, s1 = delta_tilde / delta, gamma2 / delta
        delta_tilde = alpha * c1 + gamma2 * c0 * s1
        phi, phi_tilde = c1 * phi_tilde, -s1 * phi_tilde

        if j < 2:
            P = W1 / delta
        else:
            P = (W1 + gamma1 * s_1 * P_1) / delta
        
        X = X + phi * P
        resvec = np.append(resvec, np.abs(phi_tilde))

        gamma1 = gamma2
        W0, W1 = W1, W / gamma1
        P_1, P0 = P0, P
        c0, s_1, s0 = c1, s0, s1

    return X, {'exitflag': flag, 'iters': j + 1, 'resvec': resvec}


def bls3mr(S, B, alpha, /, tol: float=1e-6, maxit: int=None, X0=None) -> tuple[np.array, dict]:
    n, p = B.shape

    if not maxit:
        maxit = n
    
    if not X0:
        X0 = np.zeros((n, p))
    X = X0
    
    if not callable(S):
        S_Mut = lambda x: S.dot(x)
    else:
        S_Mut = S
    
    R0 = B - (S_Mut(X0) + alpha * X0)
    W1, R = linalg.qr(R0, mode='economic')
    norm_r = linalg.norm(R)
    H1 = G_tilde = R
    W0 = P_1 = P0 = 0

    C_1, S_1, C0, S0 = [np.zeros((p, p)) for _ in range(4)]
    Zeros, AlphaI = np.zeros((p, p)), alpha * np.eye(p)
    

    flag, resvec = 1, np.array([norm_r])
    for j in range(maxit):
        if resvec[-1] / norm_r < tol:
            flag = 0
            break

        if j == 0:
            W = S @ W1
        else:
            W = S @ W1 + W0 @ H1.T
        W2, H2 = linalg.qr(W, mode='economic')

        if j > 1:
            T = np.vstack( (Zeros, -H1.T) )
            T = _blrotation_of_s3mr(T, C_1, S_1)
            # for i in range(p):
            #     for k in range(p):
            #         index = p + i - k - 1
            #         c, s = C_1[k, i], S_1[k, i]
            #         T[index:index+2, :] = np.array([[c, s], [-s, c]]) @ T[index:index+2, :]
            Phi, Theta = T[:p, :], T[p:, :]
        
        if j == 1:
            Theta = -H1.T

        if j > 0:
            T = np.vstack( (Theta, AlphaI) )
            T = _blrotation_of_s3mr(T, C0, S0)
            # for i in range(p):
            #     for k in range(p):
            #         index = p + i - k - 1
            #         c, s = C0[k, i], S0[k, i]
            #         T[index:index+2, :] = np.array([[c, s], [-s, c]]) @ T[index:index+2, :]
            Omga_tilde = T[p:, :]
        
        if j == 0:
            Omga_tilde = alpha * np.eye(p)
        
        C_1, S_1 = C0.copy(), S0.copy()

        T = np.vstack( (Omga_tilde, H2) )
        T1 = np.vstack( (G_tilde, Zeros) )
        T, T1 = _blrotation_of_s3mr(T, C0, S0, T1)
        # for i in range(p):
        #     for k in range(p):
        #         index = p + i - k -1

        #         delta_tilde, gamma = T[index:index+2, i]
        #         delta = np.sqrt( delta_tilde ** 2 + gamma ** 2 )
        #         c, s = delta_tilde / delta, gamma / delta
        #         C0[k, i], S0[k, i] = c, s

        #         Rot_Mat = np.array([[c, s], [-s, c]])
        #         T[index:index+2, :] = Rot_Mat @ T[index:index+2, :]
        #         T1[index:index+2, :] = Rot_Mat @ T1[index:index+2, :]

        Omega, G, G_tilde = T[:p, :], T1[:p, :], T1[p:, :]
        resvec = np.append(resvec, linalg.norm(G_tilde))

        if j < 2:
            PT = linalg.solve_triangular(Omega, W1.T, trans='T')
        else:
            PT = linalg.solve_triangular(Omega, (W1 - P_1 @ Phi).T, trans='T')

        P = PT.T
        X = X + P @ G

        W0, W1 = W1, W2
        P_1, P0 = P0, P
        H1 = H2
    
    return X, {'exitflag': flag, 'iters': j + 1, 'resvec': resvec}


def _blrotation_of_s3mr(A, C, S, /, B=None) -> np.ndarray:
    p = C.shape[0]
    for i in range(p):
        for k in range(p):
            index = p+i-k-1

            if B is not None:
                delta_tilde, gamma = A[index:index+2, i]
                delta = np.sqrt( delta_tilde ** 2 + gamma ** 2 )
                c, s = delta_tilde / delta, gamma / delta
                C[k, i], S[k, i] = c, s
            else:
                c, s = C[k, i], S[k, i]
            
            Rot_Mat = np.array([[c, s], [-s, c]])
            A[index:index+2, :] = Rot_Mat @ A[index:index+2, :]
            if B is not None:
                B[index:index+2, :] = Rot_Mat @ B[index:index+2, :]
    
    if B is not None:
        return A, B
    return A