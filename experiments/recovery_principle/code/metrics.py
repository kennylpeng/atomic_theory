"""Geometric recovery, approximate prefixes, and out-of-sample activation recovery."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching

THRESHOLDS = (.8, .9, .95, .99)
EPSILONS = (0., .05, .1)
ACT_THRESHOLDS = np.array([1e-4,.01,.03,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,1.,1.5,2.], np.float32)


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prefix(q, epsilon):
    q = np.asarray(q, bool)
    good = np.flatnonzero(np.cumsum(q) >= (1-epsilon)*np.arange(1, len(q)+1)-1e-10)
    return int(good[-1]+1) if len(good) else 0


def distinct_triple(a, b, c):
    # Hall's condition is necessary and sufficient for three left vertices.
    a,b,c = set(a),set(b),set(c)
    return bool(a and b and c and len(a|b)>=2 and len(a|c)>=2 and len(b|c)>=2 and len(a|b|c)>=3)


def geometry(A, B, kind, permutation):
    A,B = normalize(A),normalize(B)
    sim = np.abs(A @ B.T)
    out = {'atom_similarity': sim.max(1), 'atom_argmax': sim.argmax(1)}
    if kind == 'flat':
        for t in THRESHOLDS:
            out[f'flat_{t}'] = (out['atom_similarity'] >= t)[permutation]
    else:
        n = len(A)//3
        parent_score = sim[::3].max(1)
        child_score = np.zeros((n,2))
        families = {t: np.zeros(n, bool) for t in THRESHOLDS}
        for i in range(n):
            p = A[3*i]
            bp = B - (B@p)[:,None]*p
            valid = np.linalg.norm(bp,axis=1)>1e-6
            children = A[3*i+1:3*i+3]
            cp = children - (children@p)[:,None]*p
            cs = np.abs(normalize(cp) @ normalize(bp).T)
            cs[:,~valid] = 0
            child_score[i] = cs.max(1)
            for t in THRESHOLDS:
                families[t][i] = distinct_triple(np.flatnonzero(sim[3*i]>=t), np.flatnonzero(cs[0]>=t), np.flatnonzero(cs[1]>=t))
        out['parent_similarity'] = parent_score
        out['projected_child_similarity'] = child_score
        for t in THRESHOLDS:
            out[f'parent_{t}'] = (parent_score >= t)[permutation]
            out[f'child_{t}'] = (child_score >= t)[permutation]
            out[f'family_{t}'] = families[t][permutation]
    return out


def activation_selection(labels, acts):
    """All coordinate/threshold pairs, with validation-only selection.

    Sparse matrices are rows=examples, columns=true/learned features.
    Return per-pair best F1 and threshold index; no test-set optimization.
    """
    M,W = labels.shape[1],acts.shape[1]
    positives = np.asarray(labels.sum(0)).ravel().astype(np.float32)
    best = np.zeros((M,W),np.float32)
    tid = np.zeros((M,W),np.uint8)
    for h,t in enumerate(ACT_THRESHOLDS):
        pred = acts.copy()
        pred.data = (pred.data>t).astype(np.float32)
        pred.eliminate_zeros()
        pp = np.asarray(pred.sum(0)).ravel()
        tp = (labels.T@pred).toarray()
        f1 = 2*tp / np.maximum(positives[:,None]+pp[None,:],1)
        update = f1>best
        best[update]=f1[update]
        tid[update]=h
    return best,tid,positives


def evaluate_pairs(labels, acts, neurons, tids):
    y = labels.tocsc()
    z = acts.tocsc()
    f1 = np.zeros(len(neurons),np.float32)
    for i,j in enumerate(neurons):
        if j<0:
            continue
        true_rows = y.indices[y.indptr[i]:y.indptr[i+1]]
        lo,hi = z.indptr[j:j+2]
        pred_rows = z.indices[lo:hi][z.data[lo:hi]>ACT_THRESHOLDS[tids[i]]]
        tp = np.intersect1d(true_rows,pred_rows,assume_unique=True).size
        f1[i] = 2*tp/max(len(true_rows)+len(pred_rows),1)
    return f1


def activation_recovery(val_y,val_z,test_y,test_z,targets=(.9,.95)):
    best,tids,counts = activation_selection(val_y,val_z)
    neurons = best.argmax(1)
    row = np.arange(len(neurons))
    chosen = tids[row,neurons]
    out = {'activation_neuron':neurons,'activation_threshold':ACT_THRESHOLDS[chosen],
           'activation_validation_f1':best[row,neurons],
           'activation_test_f1':evaluate_pairs(test_y,test_z,neurons,chosen),
           'validation_positives':counts,'test_positives':np.asarray(test_y.sum(0)).ravel()}
    for target in targets:
        edges = sparse.csr_matrix(best>=target)
        match = maximum_bipartite_matching(edges,perm_type='column')
        chosen = tids[row,np.maximum(match,0)]
        out[f'activation_distinct_neuron_{target}'] = match
        out[f'activation_distinct_threshold_{target}'] = ACT_THRESHOLDS[chosen]
        out[f'activation_distinct_test_f1_{target}'] = evaluate_pairs(test_y,test_z,match,chosen)
    return out
