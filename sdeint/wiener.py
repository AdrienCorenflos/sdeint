# Copyright 2015 Matthew J. Aburn
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Simulation of standard multiple stochastic integrals, both Ito and Stratonovich
I_{ij}(t) = \int_{0}^{t}\int_{0}^{s} dW_i(u) dW_j(s)  (Ito)
J_{ij}(t) = \int_{0}^{t}\int_{0}^{s} \circ dW_i(u) \circ dW_j(s)  (Stratonovich)

These multiple integrals I and J are important building blocks that will be
used by most of the higher-order algorithms that integrate multi-dimensional
SODEs.

We first implement the method of Kloeden, Platen and Wright (1992) to
approximate the integrals by the first n terms from the series expansion of a
Brownian bridge process. By default using n=5.

Finally we implement the method of Wiktorsson (2001) which improves on the
previous method by also approximating the tail-sum distribution by a
multivariate normal distribution.

References:
  P. Kloeden, E. Platen and I. Wright (1992) The approximation of multiple
    stochastic integrals
  M. Wiktorsson (2001) Joint Characteristic Function and Simultaneous
    Simulation of Iterated Ito Integrals for Multiple Independent Brownian
    Motions
"""

import jax.numpy as np
import numpy as onp
from jax.lax import fori_loop, scan
from jax.random import normal, split

broadcast_to = np.broadcast_to


def deltaW(key, N, m, h):
    """Generate sequence of Wiener increments for m independent Wiener
    processes W_j(t) j=0..m-1 for each of N time intervals of length h.    

    Returns:
      dW (array of shape (N, m)): The [n, j] element has the value
      W_j((n+1)*h) - W_j(n*h) 
    """

    return np.sqrt(h) * normal(key, shape=(N, m))


def _t(a):
    """transpose the last two axes of a three axis array"""
    return a.transpose((0, 2, 1))


def _dot(a, b):
    """ for rank 3 arrays a and b, return \sum_k a_ij^k . b_ik^l (no sum on i)
    i.e. This is just normal matrix multiplication at each point on first axis
    """
    return np.matmul(a, b)


def _Aterm(key, N, h, m, k, dW):
    """kth term in the sum of Wiktorsson2001 equation (2.2)"""
    sqrt2h = np.sqrt(2.0 / h)
    key1, key2 = split(key)
    Xk = normal(key1, shape=(N, m, 1))
    Yk = normal(key2, shape=(N, m, 1))
    term1 = _dot(Xk, _t(Yk + sqrt2h * dW))
    term2 = _dot(Yk + sqrt2h * dW, _t(Xk))
    return (term1 - term2) / k


def Ikpw(key, dW, h, n=5):
    """matrix I approximating repeated Ito integrals for each of N time
    intervals, based on the method of Kloeden, Platen and Wright (1992).

    Args:
      key: jax.random.PRNGKey
      dW (array of shape (N, m)): giving m independent Weiner increments for
        each time step N. (You can make this array using sdeint.deltaW())
      h (float): the time step size
      n (int, optional): how many terms to take in the series expansion

    Returns:
      (A, I) where
        A: array of shape (N, m, m) giving the Levy areas that were used.
        I: array of shape (N, m, m) giving an m x m matrix of repeated Ito 
        integral values for each of the N time intervals.
    """
    N = dW.shape[0]
    m = dW.shape[1]
    if dW.ndim < 3:
        dW = dW.reshape((N, -1, 1))  # change to array of shape (N, m, 1)
    if dW.shape[2] != 1 or dW.ndim > 3:
        raise ValueError

    keys = split(key, n)

    def loop_body(A_i, xs):
        i, subkey = xs
        return A_i + _Aterm(subkey, N, h, m, i, dW), None

    A_init = _Aterm(keys[0], N, h, m, 1, dW)

    A, _ = scan(loop_body, A_init, [np.arange(2, n + 1, dtype=np.int32), keys[1:]])
    A = (h / (2.0 * np.pi)) * A
    I = 0.5 * (_dot(dW, _t(dW)) - np.diag(h * np.ones(m))) + A
    dW = dW.reshape((N, -1))  # change back to shape (N, m)
    return (A, I)

# The code below this point implements the method of Wiktorsson2001.

def _vec(A):
    """
    Linear operator _vec() from Wiktorsson2001 p478
    Args:
      A: a rank 3 array of shape N x m x n, giving a matrix A[j] for each
      interval of time j in 0..N-1
    Returns:
      array of shape N x mn x 1, made by stacking the columns of matrix A[j] on
      top of each other, for each j in 0..N-1
    """
    N, m, n = A.shape
    return A.reshape((N, m * n, 1), order='F')


def _unvec(vecA, m=None):
    """inverse of _vec() operator"""
    N = vecA.shape[0]
    if m is None:
        m = np.sqrt(vecA.shape[1] + 0.25).astype(np.int64)
    return vecA.reshape((N, m, -1), order='F')


def _kp(a, b):
    """Special case Kronecker tensor product of a[i] and b[i] at each
    time interval i for i = 0 .. N-1
    It is specialized for the case where both a and b are shape N x m x 1
    """
    if a.shape != b.shape or a.shape[-1] != 1:
        raise ValueError
    N = a.shape[0]
    # take the outer product over the last two axes, then reshape:
    return np.einsum('ijk,ilk->ijkl', a, b).reshape(N, -1, 1)


def _kp2(A, B):
    """Special case Kronecker tensor product of A[i] and B[i] at each
    time interval i for i = 0 .. N-1
    Specialized for the case A and B rank 3 with A.shape[0]==B.shape[0]
    """
    N = A.shape[0]
    if B.shape[0] != N:
        raise (ValueError)
    newshape1 = A.shape[1] * B.shape[1]
    return np.einsum('ijk,ilm->ijlkm', A, B).reshape(N, newshape1, -1)


def _P(m):
    """Returns m^2 x m^2 permutation matrix that swaps rows i and j where
    j = 1 + m((i - 1) mod m) + (i - 1) div m, for i = 1 .. m^2
    """
    P = onp.zeros((m ** 2, m ** 2), dtype=onp.int64)
    for i in range(1, m ** 2 + 1):
        j = 1 + m * ((i - 1) % m) + (i - 1) // m
        P[i - 1, j - 1] = 1
    return P


def _K(m):
    """ matrix K_m from Wiktorsson2001 """
    M = m * (m - 1) // 2
    K = onp.zeros((M, m ** 2), dtype=onp.int64)
    row = 0
    for j in range(1, m):
        col = (j - 1) * m + j
        s = m - j
        K[row:(row + s), col:(col + s)] = onp.eye(s)
        row += s
    return K


def _AtildeTerm(key, N, h, m, k, dW, Km0, Pm0):
    """kth term in the sum for Atilde (Wiktorsson2001 p481, 1st eqn)"""
    M = m * (m - 1) // 2
    key1, key2 = split(key)
    Xk = normal(key1, shape=(N, m, 1))
    Yk = normal(key2, shape=(N, m, 1))
    factor1 = np.dot(Km0, Pm0 - np.eye(m ** 2))
    factor1 = broadcast_to(factor1, (N, M, m ** 2))
    factor2 = _kp(Yk + np.sqrt(2.0 / h) * dW, Xk)
    return _dot(factor1, factor2) / k


def _sigmainf(N, h, m, dW, Km0, Pm0):
    """Asymptotic covariance matrix \Sigma_\infty  Wiktorsson2001 eqn (4.5)"""
    M = m * (m - 1) // 2
    Im = broadcast_to(np.eye(m), (N, m, m))
    IM = broadcast_to(np.eye(M), (N, M, M))
    Ims0 = np.eye(m ** 2)
    factor1 = broadcast_to((2.0 / h) * np.dot(Km0, Ims0 - Pm0), (N, M, m ** 2))
    factor2 = _kp2(Im, _dot(dW, _t(dW)))
    factor3 = broadcast_to(np.dot(Ims0 - Pm0, Km0.T), (N, m ** 2, M))
    return 2 * IM + _dot(_dot(factor1, factor2), factor3)


def _a(n):
    """ \sum_{n+1}^\infty 1/k^2 """
    return onp.pi ** 2 / 6.0 - sum(1.0 / k ** 2 for k in range(1, n + 1))


def Iwik(key, dW, h, n=5):
    """matrix I approximating repeated Ito integrals for each of N time
    intervals, using the method of Wiktorsson (2001).

    Args:
      key: jax.random.PRNGKey
      dW (array of shape (N, m)): giving m independent Weiner increments for
        each time step N. (You can make this array using sdeint.deltaW())
      h (float): the time step size
      n (int, optional): how many terms to take in the series expansion

    Returns:
      (Atilde, I) where
        Atilde: array of shape (N,m(m-1)//2,1) giving the area integrals used.
        I: array of shape (N, m, m) giving an m x m matrix of repeated Ito
        integral values for each of the N time intervals.
    """
    N = dW.shape[0]
    m = dW.shape[1]
    if dW.ndim < 3:
        dW = dW.reshape((N, -1, 1))  # change to array of shape (N, m, 1)
    if dW.shape[2] != 1 or dW.ndim > 3:
        raise ValueError
    if m == 1:
        return (np.zeros((N, 1, 1)), (dW * dW - h) / 2.0)
    Pm0 = _P(m)
    Km0 = _K(m)
    M = m * (m - 1) // 2

    keys = split(key, n)

    Atilde_init = _AtildeTerm(keys[0], N, h, m, 1, dW, Km0, Pm0)

    def body(Atilde, xs):
        i, subkey = xs
        return Atilde + _AtildeTerm(subkey, N, h, m, i, dW, Km0, Pm0), None

    Atilde_n, _ = scan(body, Atilde_init, [np.arange(2, n+1, dtype=np.int32), keys[1:]])

    Atilde_n = (h / (2.0 * np.pi)) * Atilde_n  # approximation after n terms
    S = _sigmainf(N, h, m, dW, Km0, Pm0)
    normdW2 = np.sum(np.abs(dW) ** 2, axis=1)
    radical = np.sqrt(1.0 + normdW2 / h).reshape((N, 1, 1))
    IM = broadcast_to(np.eye(M), (N, M, M))
    Im = broadcast_to(np.eye(m), (N, m, m))
    Ims0 = np.eye(m ** 2)
    sqrtS = (S + 2.0 * radical * IM) / (np.sqrt(2.0) * (1.0 + radical))
    G = normal(keys[-1], (N, M, 1))
    tailsum = h / (2.0 * np.pi) * _a(n) ** 0.5 * _dot(sqrtS, G)
    Atilde = Atilde_n + tailsum  # our final approximation of the areas
    factor3 = broadcast_to(np.dot(Ims0 - Pm0, Km0.T), (N, m ** 2, M))
    vecI = 0.5 * (_kp(dW, dW) - _vec(h * Im)) + _dot(factor3, Atilde)
    I = _unvec(vecI)
    dW = dW.reshape((N, -1))  # change back to shape (N, m)
    return (Atilde, I)
