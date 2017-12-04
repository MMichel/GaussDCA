#cython: cdivision=True, boundscheck=False, wraparound=False, embedsignature=True, language_level=3
from __future__ import division

cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport log, sqrt, floor

import matplotlib.pyplot as plt

ctypedef np.int8_t int8


cdef np.float64_t[:, ::1] apc_correction(const np.float64_t[:, ::1] matrix, N):
    cdef np.float64_t corr, mean
    cdef np.float64_t[::1] mean_x, mean_y
    cdef np.float64_t[:, ::1] corrected
    cdef Py_ssize_t i, j

    mean_x = np.mean(matrix, axis=0)
    mean_y = np.mean(matrix, axis=1)
    mean = np.mean(matrix)
    corrected = matrix.copy()
    for i in range(N):
        for j in range(i, N):
            corr = mean_x[j] * mean_y[i] / mean
            corrected[i, j] -= corr
            corrected[j, i] -= corr
    return corrected


cdef np.float64_t _compute_theta(int8[:, ::1] ZZ, int N, int M) nogil:
    cdef np.float64_t fracid = 0.0
    cdef np.float64_t meanfracid = 0.0

    cdef np.float64_t nids
    cdef Py_ssize_t i, j, k
    cdef int8[:] _Zi
    cdef int8[:] _Zj

    for i in range(M-1):
        _Zi = ZZ[:,i]
        for j in range(i+1,M):
            _Zj = ZZ[:,j]
            nids = 0.0
            for k in range(N):
                nids += _Zi[k] == _Zj[k]
            fracid = nids / N
            meanfracid += fracid
    meanfracid /= 0.5 * M * (M-1)
    cdef np.float64_t theta = min(0.5, 0.38 * 0.32 / meanfracid)
    return theta



cdef np.float64_t _compute_weights(int8[:, ::1] ZZ, np.float64_t theta, int N, int M,
                            np.float64_t[::1] W) nogil:
                            #np.float64_t[::1] W):
    cdef np.float64_t Meff = 0.0
    W[:] = 1.0
   
    cdef np.float64_t _thresh = floor(theta * N)

    if theta == 0:
        Meff = M
        return Meff

    cdef Py_ssize_t i, j, k
    cdef int8[:] _Zi
    cdef int8[:] _Zj
    cdef np.float64_t _dist

    for i in range(M-1):
        _Zi = ZZ[:,i]
        for j in range(i+1,M):
            _Zj = ZZ[:,j]
            _dist = 0
            k = 0
            while _dist < _thresh and k < N:
                _dist += _Zi[k] != _Zj[k]
                k += 1
            if _dist < _thresh:
                W[i] += 1
                W[j] += 1
    for i in range(M):
        W[i] = 1./W[i]
        Meff += W[i]
    return Meff



cdef void _compute_freqs(int8[:, ::1] Z, int8 q, np.float64_t[::1] W, np.float64_t Meff,
                                np.float64_t[::1] Pi,
                                np.float64_t[:, ::1] Pij,) nogil:
                                #np.float64_t[:, ::1] Pij,):
    cdef Py_ssize_t N = Z.shape[0]
    cdef Py_ssize_t M = Z.shape[1]
    cdef int8 s = q - 1

    cdef Py_ssize_t Ns = N * s

    Pij[:,:] = 0.0
    Pi[:] = 0.0

    cdef Py_ssize_t i0, j0, i, j, k
    cdef int8[::1] _Zi
    cdef int8[::1] _Zj
    cdef int8 a, b

    i0 = 0
    for i in range(N):
        _Zi = Z[i,:]
        for k in range(M):
            a = _Zi[k]
            if a == q:
                continue
            Pi[i0 + a - 1] += W[k]
        i0 += s

    i0 = 0
    for i in range(N):
        _Zi = Z[i,:]
        j0 = i0
        for j in range(i,N):
            _Zj = Z[j,:]
            for k in range(M):
                a = _Zi[k]
                b = _Zj[k]
                if a == q or b == q:
                    continue
                Pij[i0 + a - 1, j0 + b - 1] += W[k]
            j0 += s
        i0 += s

    for i in range(Ns):
        Pi[i] /= Meff
        Pij[i,i] /= Meff
        for j in range(i+1, Ns):
            Pij[i,j] /= Meff
            Pij[j,i] = Pij[i,j]

    

cdef void _compute_new_frequencies(int8[:, ::1] alignment, int8 q, np.float64_t theta,
                                    np.float64_t[::1] W,
                                    np.float64_t[::1] Pi_true, 
                                    np.float64_t[:, ::1] Pij_true, 
                                    #np.float64_t Meff) nogil:
                                    np.float64_t Meff):
    cdef Py_ssize_t N = alignment.shape[0]
    cdef Py_ssize_t M = alignment.shape[1]
    theta = _compute_theta(alignment, N, M)
    print(theta)
    print(floor(theta*N))
    Meff = _compute_weights(alignment, theta, N, M, W)
    print(Meff)
    print(np.asarray(W)[:10])
    _compute_freqs(alignment, q, W, Meff, Pi_true, Pij_true)
    print(np.asarray(Pij_true)[:10,0])

    


cdef void _add_pseudocount(np.float64_t[::1] Pi_true, np.float64_t[:, ::1] Pij_true, np.float64_t pc, int N, int8 q,
                                    np.float64_t[::1] Pi, 
                                    np.float64_t[:, ::1] Pij) nogil: 
                                    #np.float64_t[:, ::1] Pij): 
    cdef np.float64_t pcq = pc / q
    cdef Py_ssize_t i, j = 0
    cdef int8 s = q - 1
    cdef Py_ssize_t Ns = N*s

    for i in range(Ns):
        Pi[i] = (1 - pc) * Pi_true[i] + pcq
        for j in range(Ns):
            Pij[i,j] = (1 - pc) * Pij_true[i,j] + pcq / q

    cdef Py_ssize_t i0, alpha, beta, x, y

    i0 = 0
    for i in range(N):
        for alpha in range(s):
            x = i0 + alpha
            for beta in range(s):
                y = i0 + beta
                Pij[x, y] = (1 - pc) * Pij_true[x, y]
        for alpha in range(s):
            x = i0 + alpha
            Pij[x, x] += pcq
        i0 += s


cdef void _compute_FN(np.float64_t[:, ::1] mJ, int N, int8 q,
                                np.float64_t[:, ::1] mJij,
                                np.float64_t[::1] amJi,
                                np.float64_t[::1] amJj,
                                np.float64_t[:, ::1] FN) nogil:
                                #np.float64_t[:, ::1] FN):
    cdef int8 s = q - 1

    cdef Py_ssize_t i, j, a, b, _row0, _col0 = 0
    cdef np.float64_t amJ, x, fn, fn_pre
    cdef np.float64_t fs = s
    cdef np.float64_t fs2 = s*s

    for i in range(N-1):
        _row0 = i * s
        for j in range(i+1, N):
            _col0 = j * s
            amJ = 0.0
            for a in range(s):
                amJi[a] = 0.0
                amJj[a] = 0.0
            for b in range(s):
                for a in range(s):
                    x = mJ[_row0 + a, _col0 + b]
                    mJij[a,b] = x
                    amJi[b] += x / fs
                    amJj[a] += x / fs
                    amJ += x / fs2
            fn = 0.0
            for b in range(s):
                for a in range(s):
                    fn_pre = mJij[a,b] - amJi[b] - amJj[a] + amJ 
                    fn += fn_pre * fn_pre
            FN[i, j] = sqrt(fn)
            FN[j, i] = FN[i, j]


def _correct_APC(S):
    N = S.shape[0]
    Si = np.sum(S, axis=0)
    Sj = np.sum(S, axis=1)
    Sa = np.sum(S) * (1 - 1/N)
    S -= np.outer(Sj,Si) / Sa
    np.fill_diagonal(S, 0.0)
    return S


def compute_gdca_scores(alignment):
    cdef Py_ssize_t N = alignment.shape[0]
    cdef Py_ssize_t M = alignment.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] gdca

    cdef np.float64_t pseudocount = 0.8
    cdef np.float64_t theta = -1.0

    cdef int8 q = np.max(alignment)
    
    assert q < 32

    cdef int8 s = q - 1
    cdef Py_ssize_t Ns = N * s
    
    cdef np.float64_t[::1] W = np.zeros(M, dtype=np.float64)
    cdef np.float64_t[::1] Pi_true = np.zeros(Ns, dtype=np.float64)
    cdef np.float64_t[::1] Pi = np.zeros(Ns, dtype=np.float64)
    cdef np.float64_t[:, ::1] Pij_true = np.zeros((Ns, Ns), dtype=np.float64)
    cdef np.float64_t[:, ::1] Pij = np.zeros((Ns, Ns), dtype=np.float64)
    cdef np.float64_t Meff = 0.0

    _compute_new_frequencies(alignment, q, theta, W, Pi_true, Pij_true, Meff)
    _add_pseudocount(Pi_true, Pij_true, pseudocount, N, q, Pi, Pij)

    Pi_np = np.asarray(Pi)[:, np.newaxis]
    Pij_np = np.asarray(Pij)
    print(Pij_np[:10,0])
    
    C = Pij_np - (Pi_np * Pi_np.T)
    print(C[:10,0])

    #mJ_np_lower = np.linalg.inv(np.linalg.cholesky(C))
    #mJ_np = mJ_np_lower + mJ_np_lower.T - np.diag(mJ_np_lower.diagonal())
    mJ_np = np.linalg.pinv(C)
    print(mJ_np[:10,0])

    cdef np.float64_t[:, ::1] mJ = mJ_np

    cdef np.float64_t[:, ::1] mJij = np.zeros((s,s), dtype=np.float64)
    cdef np.float64_t[::1] amJi = np.zeros(s, dtype=np.float64)
    cdef np.float64_t[::1] amJj = np.zeros(s, dtype=np.float64)
    
    cdef np.float64_t[:, ::1] FN = np.zeros((N, N), dtype=np.float64)

    _compute_FN(mJ, N, q, mJij, amJi, amJj, FN)
    FN_np = np.asarray(FN, dtype=np.float64)
    print(FN_np[:10,0])
    #FN_corr = np.asarray(apc_correction(FN, N), dtype=np.float64)
    FN_corr = _correct_APC(FN_np)
    print(FN_corr[:10,0])

    results = dict(gdca=FN_np, gdca_corr=FN_corr)
    return results
