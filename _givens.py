import numpy as np
from scipy import linalg 


def sign(x):
    """
    Let x = r*e^{i*theta} belong to the complex plane, then it returns 
    e^{i*theta}. If x == 0, it returns 1.

    Parameters
    ----------
    x: float or comples

    Returns
    -------
    t: float
    a number which is x divided by norm(x). 
    """
    if x == 0:
        t = 1
    elif not isinstance(x, complex):
        t = np.sign(x)
    else:
        t = x / abs(x)
    return t


def r_givens(a, b):
    """
    Compute givens rotation G = [[c, s], [-s, c]] such that G^T * [[a], [b]] =
    [[r], [0]], where r = norm([a, b]).

    Parameters
    ----------
    a, b: float
        Input numbers.

    Returns
    -------
    c, s, r: float
        Which satisify [c, s; -s, c]^T * [a; b] = [r; 0].
    """
    if b == 0:
        if a < 0:
            c = -1
        else:
            c = 1
        s = 0
    else:
        if abs(b) > abs(a):
            tau = - a / b
            s = - np.sign(b) / np.sqrt(1 + tau**2)
            c = s * tau
        else:
            tau = - b / a
            c = np.sign(a) / np.sqrt(1 + tau**2)
            s = c * tau
    return c, s, linalg.norm([a, b])


def c_givens(u, v):
    """
    Compute givens rotation G = [[c, s], [-s.conj(), c.conj()]] 
    such that G^H * [[a], [b]] =
    [[r], [0]], where r = norm([a, b]).

    Parameters
    ----------
    a, b: complex
        Inpute numbers.

    Returns
    -------
    c, s: complex
    r: real
        Which satisify [c, s; -s', c']^H * [a; b] = [r; 0].
    """
    ru, rv = abs(u), abs(v)
    if rv == 0:
        c, s, r = u / ru, 0, ru
    else:
        c, s, r = r_givens(ru, rv)
        c = c * u / ru
        s = s * v.conjugate() / rv
    return c, s, r


def givens(a, b, iscomplex=False):
    """
    Using givens rotation such that [a; b] reduce to [r; 0].

    Parameters
    ----------
    a, b: float or complex
        Input numbers.
    iscomplex: {`False`, `True`}, optional
        Determine the real case or complex case, and `False` represents 
        complex case, `True` represents real case. Default is `False`.

    Returns
    -------
    c, s: float or complex
    r: float
        Which satisify [c, s; -s', c']^H * [a; b] = [r; 0].
    """
    if iscomplex:
        return c_givens(a, b)
    return r_givens(a, b)


def house(x, order='first'):
    """
    Compute the normalized vector v. Let H = I - 2*v*v^H, and H satisfies
    H*x = a*e_1 (e_m) where a is a scaler.

    Parameters
    ----------
    x: (m, 1) or (m, ) array like
        a column vector.
    order: {'first', 'last'}, optional
        'first' means H * x = a * e_1. 'last' means H * x = a * e_m. Default
        is 'first'.

    Return
    ------
    v: (m, 1) or (m, ) array like
        a column vector v which dimension is similar to x.
    """
    r = linalg.norm(x)
    xc = x.copy()
    if order == 'first':
        xc[0] = xc[0] + sign(xc[0]) * r;
    elif order == 'last':
        xc[-1] = xc[-1] + sign(xc[-1]) * r;

    v = xc / linalg.norm(xc)

    return v
    

def hessen_by_col(arr, cal_q=False):
    """
    Reduce a matrix to Hessenberg form started from first column.
    The Hessenberg decomposition is:
         A = Q * H * Q^H

    Parameters
    ----------
    arr: (M, M) array like
        Matrix bring to Hessenberg form.
    cal_q: bool, optional
        Whether to compute unitraty matrix. Default is False.

    Returns
    -------
    H: (M, M) arrary like
        Upper Hessenberg matrix.
    Q: (M, M) arrary like
        Unitary matrix. it will be computer if `cal_q=True`.
    """
    n = arr.shape[0]

    if cal_q:
        mq = np.eye(n)
    hessen = arr.copy()

    if n <= 2:
        if cal_q:
            return hessen, mq
        return hessen

    for i in range(n-2):
        # v = house(hessen[i+1:, i:i+1])
        v = house(hessen[i+1:, i]).reshape(-1, 1)
        vh = v.conj().T

        hessen[i+1:, i:] = hessen[i+1:, i:] - 2 * v.dot(vh.dot(hessen[i+1:, i:]))
        hessen[i+2:, i] = 0
        hessen[:, i+1:] = hessen[:, i+1:] - 2 * (hessen[:, i+1:].dot(v)).dot(vh)

        if cal_q:
            mq[1:, i+1:] = mq[1:, i+1:] - 2 * (mq[1:, i+1:].dot(v)).dot(vh)
    
    if cal_q:
        return hessen, mq
    return hessen
        

def hessen_by_row(arr, cal_q=False):
    """
    Reduce a matrix to Hessenberg form started from last row.
    The Hessenberg decomposition is:
         A = Q * H * Q^H

    Parameters
    ----------
    arr: (M, M) array like
    Matrix bring to Hessenberg form.
    cal_q: bool, optional
    Whether to compute unitraty matrix. Default is False.

    Returns
    -------
    H: (M, M) arrary like
    Upper Hessenberg matrix.
    Q: (M, M) arrary like
    Unitary matrix. it will be computer if `cal_q=True`.
    """
    n = arr.shape[0]

    if cal_q:
        q = np.eye(n)
    h = arr.copy()

    if n <= 2:
        if cal_q:
            return h, q
        return h
    
    for i in range(n-1, 1, -1):
        # v = house(h[i:i+1, :i].T, order='last')
        v = house(h[i, :i], order='last').reshape(-1, 1)
        vh = v.conj().T

        h[:i+1, :i] = h[:i+1, :i] - 2 * (h[:i+1, :i].dot(v)).dot(vh)
        h[i, :i-1] = 0
        h[:i, :] = h[:i, :] - 2 * v.dot(vh.dot(h[:i, :]))

        if cal_q:
            q[:n-1, :i] = q[:n-1, :i] - 2 * (q[:n-1, :i].dot(v)).dot(vh)
    
    if cal_q:
        return h, q
    return h
        

def hessen(arr, order=0, cal_q=False):
    """
    Reduce a matrix to Hessenberg form.
    The Hessenberg decomposition is:
         A = Q * H * Q^H

    Parameters
    ----------
    arr: (M, M) array like
        Matrix bring to Hessenberg form.
    order: {0, 1}, optional
        Determine how to reduce a matrix to a Hessenberg form. 
        0 means that start by first column, 1 means that start by last row.
        Default is 0.
    cal_q: bool, optional
        Whether to compute unitraty matrix. Default is False.

    Returns
    -------
    H: (M, M) arrary like
    Upper Hessenberg matrix.
    Q: (M, M) arrary like
    Unitary matrix. it will be computer if `cal_q=True`.
    """
    if order == 1:
        return hessen_by_row(arr, cal_q)
    return hessen_by_col(arr, cal_q)


if __name__ == '__main__':
    # linalg.hessenberg
    # np.reshape
    m = 100
    A = np.random.randn(m, m)
    H, Q = hessen(A, cal_q=True, order=1)
    delA = A - Q.dot(H.dot(Q.conj().T))
    # print(H, '\n', Q, '\n')
    # print(linalg.norm(delA, ord=2))
    print(np.allclose(delA, np.zeros(delA.shape)))


