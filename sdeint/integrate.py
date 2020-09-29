# Copyright 2015 Matthew J. Aburn
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""Numerical integration algorithms for Ito and Stratonovich stochastic
ordinary differential equations.

Usage:
    itoint(f, G, y0, tspan)  for Ito equation dy = f dt + G dW
    stratint(f, G, y0, tspan)  for Stratonovich equation dy = f dt + G \circ dW

    y0 is the initial value
    tspan is an array of time values (currently these must be equally spaced)
    function f is the deterministic part of the system (scalar or  dx1  vector)
    function G is the stochastic part of the system (scalar or  d x m matrix)

sdeint will choose an algorithm for you. Or you can choose one explicitly:

itoEuler: the Euler-Maruyama algorithm for Ito equations.
stratHeun: the Stratonovich Heun algorithm for Stratonovich equations.
itoSRI2: the Roessler2010 order 1.0 strong Stochastic Runge-Kutta
  algorithm SRI2 for Ito equations.
stratSRS2: the Roessler2010 order 1.0 strong Stochastic Runge-Kutta
  algorithm SRS2 for Stratonovich equations.
stratKP2iS: the Kloeden and Platen two-step implicit order 1.0 strong algorithm
  for Stratonovich equations.
"""

from __future__ import absolute_import

import jax.numpy as np
from jax import vmap
from jax.lax import scan
from jax.numpy import vectorize
from jax.random import split

from .wiener import deltaW, Ikpw


class Error(Exception):
    pass


class SDEValueError(Error):
    """Thrown if integration arguments fail some basic sanity checks"""
    pass


def itoint(key, f, G, y0, tspan, m):
    """ Numerically integrate the Ito equation  dy = f(y,t)dt + G(y,t)dW

    where y is the d-dimensional state vector, f is a vector-valued function,
    G is an d x m matrix-valued function giving the noise coefficients and
    dW(t) = (dW_1, dW_2, ... dW_m) is a vector of independent Wiener increments

    Args:
      f: callable(y,t) returning a numpy array of shape (d,)
         Vector-valued function to define the deterministic part of the system
      G: callable(y,t) returning a numpy array of shape (d,m)
         Matrix-valued function to define the noise coefficients of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan: (array) The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
      m: int
        dimension of the latent brownian

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row

    Raises:
      SDEValueError
    """
    # In future versions we can automatically choose here the most suitable
    # Ito algorithm based on properties of the system and noise.
    chosenAlgorithm = itoSRI2
    return chosenAlgorithm(key, f, G, np.atleast_1d(y0), tspan, m)


def itoEuler(key, f, G, y0, tspan, m):
    """Use the Euler-Maruyama algorithm to integrate the Ito equation
    dy = f(y,t)dt + G(y,t) dW(t)

    where y is the d-dimensional state vector, f is a vector-valued function,
    G is an d x m matrix-valued function giving the noise coefficients and
    dW(t) = (dW_1, dW_2, ... dW_m) is a vector of independent Wiener increments

    Args:
      key: jax.random.PRNGKey
      f: callable(y, t) returning (d,) array
         Vector-valued function to define the deterministic part of the system
      G: callable(y, t) returning (d,m) array
         Matrix-valued function to define the noise coefficients of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan: (array) The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
      m: int
        dimension of the latent brownian

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row

    Raises:
      SDEValueError

    See also:
      G. Maruyama (1955) Continuous Markov processes and stochastic equations
      Kloeden and Platen (1999) Numerical Solution of Differential Equations
    """
    N = tspan.shape[0]
    h = (tspan[N - 1] - tspan[0]) / (N - 1)
    # allocate space for result
    # y = np.zeros((N, d), dtype=type(y0[0]))
    dWs = deltaW(key, N, m, h)

    def body(yn, xs):
        dW, tn = xs
        ynp1 = yn + f(yn, tn) * h + np.dot(G(yn, tn), dW)
        return ynp1, yn

    _, ys = scan(body, np.atleast_1d(y0), (dWs, tspan))

    return ys


def itoSRI2(key, f, G, y0, tspan, m, Imethod=Ikpw):
    """Use the Roessler2010 order 1.0 strong Stochastic Runge-Kutta algorithm
    SRI2 to integrate an Ito equation dy = f(y,t)dt + G(y,t)dW(t)

    where y is d-dimensional vector variable, f is a vector-valued function,
    G is a d x m matrix-valued function giving the noise coefficients and
    dW(t) is a vector of m independent Wiener increments.

    This algorithm is suitable for Ito systems with an arbitrary noise
    coefficient matrix G (i.e. the noise does not need to be scalar, diagonal,
    or commutative). The algorithm has order 2.0 convergence for the
    deterministic part alone and order 1.0 strong convergence for the complete
    stochastic system.

    Args:
      key: jax.random.PRNGKey
      f: A function f(y, t) returning an array of shape (d,)
         Vector-valued function to define the deterministic part of the system

      G: The d x m coefficient function G can be given in two different ways:

         You can provide a single function G(y, t) that returns an array of
         shape (d, m). In this case the entire matrix G() will be evaluated
         2m+1 times at each time step so complexity grows quadratically with m.

         Alternatively you can provide a list of m functions g(y, t) each
         defining one column of G (each returning an array of shape (d,).
         In this case each g will be evaluated 3 times at each time step so
         complexity grows linearly with m. If your system has large m and
         G involves complicated functions, consider using this way.

      y0: array of shape (d,) giving the initial state vector y(t==0)

      tspan: (array) The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.

      m: int
        dimension of the latent brownian

      Imethod (callable, optional): which function to use to simulate repeated
        Ito integrals. Here you can choose either sdeint.Ikpw (the default) or
        sdeint.Iwik (which is more accurate but uses a lot of memory in the
        current implementation).

      dW: optional array of shape (len(tspan)-1, d).
      I: optional array of shape (len(tspan)-1, m, m).
        These optional arguments dW and I are for advanced use, if you want to
        use a specific realization of the d independent Wiener processes and
        their multiple integrals at each time step. If not provided, suitable
        values will be generated randomly.

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row

    Raises:
      SDEValueError

    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    return _Roessler2010_SRK2(key, f, G, y0, tspan, Imethod, m)


def _Roessler2010_SRK2(key, f, G, y0, tspan, IJmethod, m):
    """Implements the Roessler2010 order 1.0 strong Stochastic Runge-Kutta
    algorithms SRI2 (for Ito equations) and SRS2 (for Stratonovich equations).

    Algorithms SRI2 and SRS2 are almost identical and have the same extended
    Butcher tableaus. The difference is that Ito repeateded integrals I_ij are
    replaced by Stratonovich repeated integrals J_ij when integrating a
    Stratonovich equation (Theorem 6.2 in Roessler2010).

    Args:
      key: jax.random.PRNGKey
      f: A function f(y, t) returning an array of shape (d,)
      G: Either a function G(y, t) that returns an array of shape (d, m),
         or a list of m functions g(y, t) each returning an array shape (d,).
      y0: array of shape (d,) giving the initial state
      tspan : (array) Sequence of equally spaced time points
      IJmethod (callable): which function to use to generate repeated
        integrals. N.B. for an Ito equation, must use an Ito version here
        (either Ikpw or Iwik). For a Stratonovich equation, must use a
        Stratonovich version here (Jkpw or Jwik).
      m: int
        dimension of the latent brownian
      dW: optional array of shape (len(tspan)-1, d).
      IJ: optional array of shape (len(tspan)-1, m, m).
        Optional arguments dW and IJ are for advanced use, if you want to
        use a specific realization of the d independent Wiener processes and
        their multiple integrals at each time step. If not provided, suitable
        values will be generated randomly.

    Returns:
      y: array, with shape (len(tspan), len(y0))

    Raises:
      SDEValueError

    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    N = len(tspan)
    h = (tspan[N - 1] - tspan[0]) / (N - 1)  # assuming equal time steps
    key1, key2 = split(key, 2)
    dWs = deltaW(key1, N - 1, m, h)  # shape (N, m)
    __, I = IJmethod(key2, dWs, h)  # shape (N, m, m)

    vecG = vectorize(G, excluded=(1,), signature="(d)->(m)")

    def body(yn, xs):
        tn, tnp1, Ik, Iij = xs
        h = tnp1 - tn
        sqrth = np.sqrt(h)
        fnh = f(yn, tn) * h
        Gn = G(yn, tn)
        sum1 = np.dot(Gn, Iij) / sqrth
        H20 = (yn + fnh)
        H20b = (yn + fnh).reshape(-1, 1)
        H2 = H20b + sum1
        H3 = H20b - sum1
        fn1h = f(H20, tnp1) * h

        ynp1 = yn + 0.5 * (fnh + fn1h) + np.dot(Gn, Ik)

        ynp1 = ynp1 + 0.5 * sqrth * np.sum(vecG(H2.T, tnp1) - vecG(H3.T, tnp1), axis=0)

        return ynp1, yn

    _, ys = scan(body, y0, (tspan[:-1], tspan[1:], dWs, I))

    return ys
