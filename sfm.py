"""
minimization of submodular functions


to start:
minimization of symmetric submodular set functions using Queyranne's algorithm

Maurice Queyranne. 1998. Minimizing symmetric submodular functions. Mathematical
Programming.


Jason Manley, 2018
"""

import numpy as np

def pendent_pair(Vprime, V, S, f, params=None):
    """ return Queyranne's pendent pair for function f over finite set V,
        where the first (arbitary) input element x is V[0] """
    x = 0
    vnew = Vprime[x]
    n = len(Vprime)
    Wi = []
    used = np.zeros((n,1))
    used[x] = 1

    for i in range(n-1):
        vold = vnew
        Wi = Wi + S[vold]

        ## update keys
        keys = np.ones((n,1))*np.inf
        for j in range(n):
            if used[j]:
                continue
            keys[j] = f(Wi + S[Vprime[j]], V, params) - f(S[Vprime[j]], V, params)

        ## extract min
        argmin = np.argmin(keys)
        vnew = Vprime[argmin]
        used[argmin] = 1
        fval = np.min(keys)

    s = vold
    t = vnew

    return s, t, fval


def diff(A, B):
    """ find set difference (relative complement, or A\B) """
    m = np.amax(np.array([np.amax(A), np.amax(B)]))
    vals = np.zeros((m+1,1))
    vals[A] = 1
    vals[B] = 0
    idx = np.nonzero(vals)
    return idx[0]


def optimal_set(V, f, params=None):
    """ implement Queyranne's algorithm for finding the minimum of symmetric
        submodular function f over the finite set V

        to utilize: define a function f(S, V, params). Queyranne's algorithm will
        minimize f[S] + f[V\S] given any necessary parameters in params """

    n = len(V)
    S = [[] for _ in range(n)]
    for i in range(n):
        S[i] = [V[i]]

    p = np.zeros((n-1,1))
    A = []
    idxs = range(n)
    for i in range(n-1):
        ## find a pendant pair
        t, u, fval = pendent_pair(idxs, V, S, f, params)

        ## candidate solution
        A.append(S[u])
        p[i] = f(S[u],V,params)
        S[t] = [*S[t], *S[u]]
        idxs = diff(idxs, u)
        S[u] = []

    ## return minimum solution
    i = np.argmin(p)
    R = A[i]
    fval = p[i]

    ## make R look pretty
    notR = diff(V,R)
    R = sorted(R)
    notR = sorted(notR)

    if R[0] < notR[0]:
        R = (tuple(R),tuple(notR))
    else:
        R = (tuple(notR),tuple(R))

    return R, fval


def k_subset(s, k):
    if k == len(s):
        return (tuple([(x,) for x in s]),)
    k_subs = []
    for i in range(len(s)):
        partials = k_subset(s[0:i] + s[i + 1:len(s)], k)
        for partial in partials:
            for p in range(len(partial)):
                k_subs.append(partial[:p] + (partial[p] + (s[i],),) + partial[p + 1:])
    return k_subs


def uniq_subsets(s):
    u = set()
    for x in s:
        t = []
        for y in x:
            y = list(y)
            y.sort()
            t.append(tuple(y))
        t.sort()
        u.add(tuple(t))
    return u


def find_partitions(V,k):
    """ find all partitions of elements in V that contain k members """
    k_subs = k_subset(V,k)
    k_subs = uniq_subsets(k_subs)

    return k_subs


def intersection(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))


def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))


if __name__=='__main__':
    print('Tests to come :)')
