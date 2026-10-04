import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("name", ["scripts/stability/persistent_matching.py", "src/atomic_features/matching.py"])
def test_full_source_matching_before_intersection(name):
    spec = importlib.util.spec_from_file_location("matching_test", ROOT/name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    graphs = [csr_matrix([[1,0],[1,0]]), csr_matrix([[0,0,0],[1,0,0]])]
    selected, destinations, stats = m.persistent_matching(graphs)
    assert len(selected) == 0
    assert stats["pairwise_counts"] == [1,1]
    assert stats["method"] == "independent_pairwise_intersection"
    one = m.persistent_matching(graphs[:1])
    np.testing.assert_array_equal(one[0], [0])
    m.validate_witness(graphs, selected, destinations)

@pytest.mark.parametrize("name", ["scripts/stability/persistent_matching.py", "src/atomic_features/matching.py"])
def test_nonmaximum_pairwise_assignment_rejected(name):
    spec=importlib.util.spec_from_file_location("matching_test", ROOT/name)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with pytest.raises(ValueError):
        m.certify_pairwise_maximum(csr_matrix(np.eye(2)), np.array([0,-1]))
