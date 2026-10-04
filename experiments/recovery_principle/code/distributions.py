"""Exact product-weight fixed-cardinality sampling; no sequential PPS approximation."""
from dataclasses import dataclass, asdict
import numpy as np
import torch


class ProductSubset:
    """P(S) = prod(w[S]) / e_k(w), |S|=k, to float64 accuracy.

    E[r,j] is log e_r(w[j:]). Given next index >=s, its survival
    function at j is exp(E[r,j]-E[r,s]). Invert it with searchsorted.
    Preprocessing O(n*k); independent samples O(batch*k*log(n)).
    """
    def __init__(self, n, k, alpha, allow_outside_theorem=False):
        if not 0 < k <= n or not np.isfinite(alpha) or (alpha <= 1 and not allow_outside_theorem):
            raise ValueError('Require 0 < k <= n and alpha > 1')
        self.n, self.k = n, k
        self.logw = -alpha * np.log(np.arange(1, n+1, dtype=np.float64))
        self.E = np.full((k+1, n+1), -np.inf)
        self.E[0] = 0
        for j in range(n-1, -1, -1):
            self.E[1:, j] = np.logaddexp(self.E[1:, j+1], self.logw[j]+self.E[:-1, j+1])

    def sample(self, rng, size):
        start = np.zeros(size, dtype=np.int64)
        out = np.empty((size, self.k), dtype=np.int64)
        for slot, remaining in enumerate(range(self.k, 0, -1)):
            u = np.maximum(rng.random(size), np.finfo(float).tiny)
            target = self.E[remaining, start] + np.log(u)
            idx = np.searchsorted(-self.E[remaining], -target, side='right') - 1
            idx = np.minimum(idx, self.n-remaining)
            out[:, slot] = idx
            start = idx+1
        return out


@dataclass(frozen=True)
class Spec:
    kind: str = 'flat'
    M: int = 4096
    d: int = 512
    K: int = 8
    alpha: float = 1.5
    dictionary_seed: int = 100
    dictionary: str = 'gaussian'
    permutation: str = 'identity'
    permutation_seed: int = 700
    allow_outside_theorem: bool = False


class Distribution:
    def __init__(self, spec, device='cpu'):
        self.spec = spec
        if spec.kind not in ('flat', 'hierarchical'):
            raise ValueError(spec.kind)
        if spec.kind == 'hierarchical' and (spec.M % 3 or spec.K % 2):
            raise ValueError('Hierarchical model requires M=3N and K=2L')
        n = spec.M if spec.kind == 'flat' else spec.M//3
        k = spec.K if spec.kind == 'flat' else spec.K//2
        self.sampler = ProductSubset(n, k, spec.alpha, spec.allow_outside_theorem)
        self.permutation = np.arange(n)
        prng = np.random.default_rng(spec.permutation_seed)
        if spec.permutation == 'random':
            self.permutation = prng.permutation(n)
        elif spec.permutation == 'reverse':
            self.permutation = self.permutation[::-1].copy()
        elif spec.permutation == 'half_shared':
            # Preserve the first n/8 ranks; permute everything else.
            cut = max(1, n//8)
            self.permutation[cut:] = prng.permutation(self.permutation[cut:])
        elif spec.permutation != 'identity':
            raise ValueError(spec.permutation)
        rng = np.random.default_rng(spec.dictionary_seed)
        if spec.dictionary == 'orthogonal':
            if spec.d < spec.M:
                raise ValueError('Orthogonal dictionary requires d >= M')
            A = np.linalg.qr(rng.standard_normal((spec.d, spec.M)))[0].T
        elif spec.dictionary == 'gaussian':
            A = rng.standard_normal((spec.M, spec.d))
            A /= np.linalg.norm(A, axis=1, keepdims=True)
        else:
            raise ValueError(spec.dictionary)
        self.A = torch.tensor(A, dtype=torch.float32, device=device)

    def latent(self, rng, size):
        selected = self.permutation[self.sampler.sample(rng, size)]
        if self.spec.kind == 'hierarchical':
            # Interleaved [parent, child 1, child 2], exactly as in the paper.
            parents = 3*selected
            children = parents + 1 + rng.integers(0, 2, selected.shape)
            ids = np.concatenate([parents, children], axis=1)
        else:
            ids = selected
        values = rng.random(ids.shape, dtype=np.float32)
        # Exclude the finite-precision endpoint 0 (continuous uniform has no atom there).
        values = np.maximum(values, np.nextafter(np.float32(0), np.float32(1)))
        return ids, values

    def batch(self, rng, size):
        ids, values = self.latent(rng, size)
        it = torch.as_tensor(ids, device=self.A.device)
        vt = torch.as_tensor(values, device=self.A.device)
        x = (self.A[it] * vt[..., None]).sum(1)
        return x, ids, values
