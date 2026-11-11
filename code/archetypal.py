"""A small, dependency-light implementation of Archetypal Analysis (AA).

Model (Cutler & Breiman, 1994):   X  ~=  S @ C @ X  =  S @ A
    S (N x K): each row is a convex combination over the K archetypes
    C (K x N): each row is a convex combination over the N data points
    A = C @ X: the archetypes themselves

Both S and C are fitted by alternating accelerated projected-gradient (FISTA)
steps onto the probability simplex, so every update costs O(N*M*K), which puts
it in the same family as the PCHA solver of Moerup & Hansen (2012).
"""
import numpy as np


# --------------------------------------------------------------------------- #
# Simplex helpers
# --------------------------------------------------------------------------- #
def project_simplex(V):
    """Project each row of V onto the probability simplex (Duchi et al., 2008)."""
    n, k = V.shape
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, k + 1)
    cond = U - css / ind > 0
    rho = k - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(n), rho] / (rho + 1.0)
    return np.maximum(V - theta[:, None], 0.0)


def _fista_rows(grad, L, Z0, iters):
    """Minimise a smooth function over row-stochastic matrices with FISTA."""
    Z, Y, t = Z0.copy(), Z0.copy(), 1.0
    for _ in range(iters):
        Z_new = project_simplex(Y - grad(Y) / L)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y = Z_new + ((t - 1.0) / t_new) * (Z_new - Z)
        Z, t = Z_new, t_new
    return Z


def solve_weights(X, A, S0=None, iters=100):
    """S = argmin ||X - S A||_F^2  subject to every row of S lying on the simplex."""
    N, K = X.shape[0], A.shape[0]
    S0 = np.full((N, K), 1.0 / K) if S0 is None else S0
    AAt, XAt = A @ A.T, X @ A.T
    L = np.linalg.norm(AAt, 2) + 1e-12
    return _fista_rows(lambda S: S @ AAt - XAt, L, S0, iters)


def distance_to_hull(X, A, iters=150):
    """Squared distance from each row of X to conv(A)."""
    S = solve_weights(X, A, iters=iters)
    return np.sum((X - S @ A) ** 2, axis=1)


# --------------------------------------------------------------------------- #
# Initialisation strategies (Section VI-A of the survey)
# --------------------------------------------------------------------------- #
def init_indices(X, K, method, rng):
    N = X.shape[0]
    if method == "uniform":
        return rng.choice(N, K, replace=False)
    first = int(rng.integers(N))
    idx = [first]
    if method == "furthestsum":
        dsum = np.linalg.norm(X - X[first], axis=1)
        for _ in range(K - 1):
            d = dsum.copy()
            d[idx] = -np.inf
            nxt = int(np.argmax(d))
            idx.append(nxt)
            dsum += np.linalg.norm(X - X[nxt], axis=1)
        return np.array(idx)
    if method == "aa++":
        for _ in range(K - 1):
            d = distance_to_hull(X, X[idx])
            d[idx] = 0.0
            p = d / d.sum()
            idx.append(int(rng.choice(N, p=p)))
        return np.array(idx)
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_aa(X, K, init="furthestsum", n_iter=300, inner=25, tol=1e-6, seed=0):
    """Fit AA and return S, C, A, the final RSS and the RSS history."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = init_indices(X, K, init, rng)
    C = np.zeros((K, N))
    C[np.arange(K), idx] = 1.0
    A = C @ X
    S = solve_weights(X, A, iters=200)
    X2 = np.linalg.norm(X, 2) ** 2
    history = [np.sum((X - S @ A) ** 2)]

    for _ in range(n_iter):
        StS, StX = S.T @ S, S.T @ X
        L = np.linalg.norm(StS, 2) * X2 + 1e-12
        C = _fista_rows(lambda Cm: (StS @ (Cm @ X) - StX) @ X.T, L, C, inner)
        A = C @ X
        S = solve_weights(X, A, S0=S, iters=inner)
        rss = np.sum((X - S @ A) ** 2)
        history.append(rss)
        if history[-2] - rss < tol * history[-2]:
            break

    return dict(S=S, C=C, A=A, rss=history[-1], history=np.array(history), init_idx=idx)


# --------------------------------------------------------------------------- #
# Diagnostics (Section VI-C/D of the survey)
# --------------------------------------------------------------------------- #
def variance_explained(X, rss):
    return 1.0 - rss / np.sum((X - X.mean(0)) ** 2)


def nmi(S1, S2):
    """Normalised mutual information between two soft assignment matrices."""
    def mi(P, Q):
        J = P.T @ Q / P.shape[0]
        J = np.clip(J, 1e-300, None)
        return np.sum(J * np.log(J / (J.sum(1, keepdims=True) * J.sum(0, keepdims=True))))
    return 2 * mi(S1, S2) / (mi(S1, S1) + mi(S2, S2))


def archetype_similarity(A1, A2, X):
    """sim(A, A') = 1 - mean matched squared distance / total feature variance."""
    from scipy.optimize import linear_sum_assignment
    D = ((A1[:, None, :] - A2[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(D)
    return 1.0 - D[r, c].mean() / X.var(0).sum()
