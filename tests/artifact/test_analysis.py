import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from atomic_features.matching import (
    persistent_matching,
    validate_witness,
    cosine_graph,
    candidate_graph,
)
from atomic_features.probes import fit_probe, evaluate_probe, Probe
from atomic_features.sae import SAE


def test_pairwise_optima_are_not_persistence():
    graphs = [
        csr_matrix(a)
        for a in (
            [[1, 0], [1, 0], [0, 1]],
            [[0, 1], [1, 0], [1, 0]],
            [[1, 0], [0, 1], [1, 0]],
        )
    ]
    assert all(
        np.count_nonzero(maximum_bipartite_matching(g, perm_type="column") >= 0) == 2
        for g in graphs
    )
    selected, dst, stats = persistent_matching(graphs)
    assert len(selected) == 1 and stats["method"] == "independent_pairwise_intersection"
    validate_witness(graphs, selected, dst)


def test_persistence_is_intersection_of_full_graph_assignments():
    rng = np.random.default_rng(731)
    for _ in range(35):
        graphs = [csr_matrix(rng.random((6, 4)) < 0.35) for _ in range(3)]
        assignments = [maximum_bipartite_matching(g, perm_type="column") for g in graphs]
        expected = np.flatnonzero(np.logical_and.reduce([a >= 0 for a in assignments]))
        selected, dst, _ = persistent_matching(graphs)
        np.testing.assert_array_equal(selected, expected)
        for actual, assignment in zip(dst, assignments):
            np.testing.assert_array_equal(actual, assignment[selected])
        validate_witness(graphs, selected, dst)


def test_empty_support_and_single_layer():
    for graphs, expected in [
        ([csr_matrix([[1, 0], [1, 0], [0, 1]])], 2),
        ([csr_matrix([[1], [0]]), csr_matrix([[0], [1]])], 0),
    ]:
        selected, dst, _ = persistent_matching(graphs)
        assert len(selected) == expected
        validate_witness(graphs, selected, dst)


@pytest.mark.parametrize(
    "source,destination",
    [([-1], [0]), ([0], [2]), ([0, 0], [0, 1]), ([0], [1]), ([0.5], [0])],
)
def test_bad_witnesses_fail_without_python_asserts(source, destination):
    with pytest.raises(ValueError):
        validate_witness(
            [csr_matrix(np.eye(2))], np.array(source), [np.array(destination)]
        )


def test_signed_cosine_and_threshold_boundary():
    a = np.array([[0.7, np.sqrt(0.51)], [-1, 0], [0, 1]], dtype=np.float32)
    b = np.eye(2, dtype=np.float32)
    signed = cosine_graph(a, b, batch_size=1).toarray()
    assert signed[0, 0] and not signed[1, 0]
    assert cosine_graph(a, b, absolute=True, batch_size=2).toarray()[1, 0]
    np.testing.assert_array_equal(signed, (a @ b.T) >= np.float32(0.7))


def test_candidate_union_signed_finite_and_inclusive():
    f = np.array([[0, 1], [0, -1]])
    fs = np.array([[0.7, -0.9], [np.nan, 0.9]], dtype=np.float32)
    r = np.array([[1], [0]])
    rs = np.array([[0.8], [0.6]])
    np.testing.assert_array_equal(
        candidate_graph(f, fs, r, rs).toarray(), [[1, 0], [1, 0]]
    )


def test_probe_threshold_ties_and_heldout_metrics():
    x = np.array([[1, 1], [1, 1], [0.5, 0.5], [0, 0]])
    probe = fit_probe(x, [1, 0, 1, 0])
    assert probe == Probe(0, 0.5)
    metrics = evaluate_probe(probe, [[0.5, 0], [0.49, 0], [0, 0]], [1, 1, 0])
    assert (
        metrics["tp"] == 1
        and metrics["fn"] == 1
        and metrics["f1"] == pytest.approx(2 / 3)
    )
    with pytest.raises(ValueError):
        fit_probe(x, [0, 0, 0, 0])


def test_probe_matches_bruteforce_threshold_oracle():
    rng = np.random.default_rng(99)
    for _ in range(25):
        x = rng.integers(0, 5, (20, 6)).astype(np.float32)
        y = rng.random(20) < 0.4
        probe = fit_probe(x, y)
        candidates = []
        for f in range(x.shape[1]):
            best = None
            for threshold in sorted(set(x[:, f]) - {0}, reverse=True):
                stats = evaluate_probe(Probe(f, float(threshold)), x, y)
                if stats["tp"] < 1:
                    continue
                if best is None or stats["f1"] > best[0]:
                    best = (stats["f1"], stats["tp"], -stats["fp"], threshold)
            if best:
                candidates.append((best[:3], -f, best[3]))
        winner = max(candidates)
        assert probe == Probe(-winner[1], winner[2])


def test_sae_uses_centering_not_encoder_bias_and_preserves_ids(tmp_path):
    enc = np.array([[1, -1, 0], [0, 0, 2]], dtype=np.float32)
    dec = np.array([[1, 0], [-1, 0], [0, 1]], dtype=np.float32)
    bias = np.array([2, 3], dtype=np.float32)
    for name, a in [
        ("W_enc", enc),
        ("W_dec", dec),
        ("b_dec", bias),
        ("b_enc", np.full(3, 100, dtype=np.float32)),
    ]:
        np.save(tmp_path / f"{name}.npy", a)
    (tmp_path / "model.json").write_text(
        json.dumps(dict(top_k=1, input_unit_norm=False, model_type="topk"))
    )
    model = SAE(tmp_path)
    x = np.array([[4, 3], [1, 3], [2, 5], [2, 3]], dtype=np.float32)
    acts = model.encode(x, batch_size=2)
    np.testing.assert_array_equal(
        acts.toarray(), [[2, 0, 0], [0, 1, 0], [0, 0, 4], [0, 0, 0]]
    )
    np.testing.assert_array_equal(
        model.reconstruct(acts), [[4, 3], [1, 3], [2, 7], [2, 3]]
    )
