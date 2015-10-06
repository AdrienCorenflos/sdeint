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

import numpy as np


def deltaW(N, m, delta_t):
    """Generate sequence of Wiener increments for each of m independent
    Wiener processes W_j(t) j=1..m over N time steps with constant
    time step size delta_t.
    
    Returns:
      array of shape (N, m), where the [k, j] element has the value
        W_j((k+1)delta_t) - W_j(k*delta_t)
    """
    return np.random.normal(0.0, np.sqrt(delta_t), (N, m))


def _t(a):
    """transpose the last two axes of a three axis array"""
    return a.transpose((0, 2, 1))


def _dot(a, b):
    """ for rank 3 arrays a and b, return \sum_k a_ij^k . b_ik^l (no sum on i)
    i.e. This is just normal matrix multiplication at each point on first axis
    """
    return np.einsum('ijk,ikl->ijl', a, b)


def _Aterm(N, h, m, k, dW):
    """kth term in the sum of Wiktorsson2001 equation (2.2)"""
    sqrt2h = np.sqrt(2.0/h)
    Xk = np.random.normal(0.0, 1.0, (N, m, 1))
    Yk = np.random.normal(0.0, 1.0, (N, m, 1))
    term1 = _dot(Xk, _t(Yk + sqrt2h*dW))
    term2 = _dot(Yk + sqrt2h*dW, _t(Xk))
    return (term1 - term2)/k


def Ikpw(N, h, m, n=5):
    """
    matrix I approximating repeated Ito integrals at each of N time intervals,
    based on the method of Kloeden, Platen and Wright (1992).

    Args:
      N (int): the number of time intervals
      h (float): the time step size
      m (int): the number of independent Wiener processes
      n (int, optional): how many terms to take in the series expansion

    Returns:
      (dW, A, I) where
        I: array of shape (N, m, m) giving our approximation of the m x m 
          matrix of repeated Ito integrals for each of N time intervals.
        A: array of shape (N, m, m) giving the Levy areas that were used.
        dW: array of shape (N, m, 1) giving the m Wiener increments at each
          time interval.
    """
    dW = deltaW(N, m, h)
    dW = np.expand_dims(dW, -1) # array of shape N x m x 1
    A = _Aterm(N, h, m, 1, dW)
    for k in range(2, n+1):
        A += _Aterm(N, h, m, k, dW)
    A = (h/(2.0*np.pi))*A
    I = 0.5*(_dot(dW, _t(dW)) - np.diag(h*np.ones(m))) + A
    return (dW, A, I)


def _vec(A):
    """For each time interval j, stack columns of matrix A[j] on top of each
    other to give a long vector.
    That is, if A is N x m x n then _vec(A) will be N x mn x 1
    """
    N, m, n = A.shape
    return A.reshape((N, m*n, 1), order='F')

def _kp(a, b):
    """Special case Kronecker tensor product of a[i] and b[i] at each 
    time interval i for i = 0 .. N-1
    It is specialized for the case where both a and b are shape N x m x 1 
    """
    if a.shape != b.shape or a.shape[-1] != 1:
        raise(ValueError)
    N = a.shape[0]
    # take the outer product over the last two axes, then reshape:
    return np.einsum('ijk,ilk->ijkl', a, b).reshape(N, -1, 1)


def _P(m):
    """Returns m^2 x m^2 permutation matrix that swaps rows i and j where
    j = 1 + m((i - 1) mod m) + (i - 1) div m, for i = 1 .. m^2
    """
    P = np.zeros((m**2,m**2), dtype=np.int64)
    for i in range(1, m**2 + 1):
        j = 1 + m*((i - 1) % m) + (i - 1)//m
        P[i-1, j-1] = 1
    return P


def _K(m):

    pass


def Jkpw(h, m, n=5):
    """
    matrix J approximating repeated Ito integrals at each of N time intervals,
    based on the method of Kloeden, Platen and Wright (1992).

    Args:
      N (int): the number of time intervals
      h (float): the time step size
      m (int): the number of independent Wiener processes
      n (int, optional): how many terms to take in the series expansion

    Returns:
      (dW, A, J) where
        J: array of shape (N, m, m) giving our approximation of the m x m 
          matrix of repeated Ito integrals for each of N time intervals.
        A: array of shape (N, m, m) giving the Levy areas that were used.
        dW: array of shape (N, m, 1) giving the m Wiener increments at each
          time interval.
    """
    pass


def Iwiktorsson(h, m, n=5):
    pass


def Jwiktorsson(h, m, n=5):
    pass
