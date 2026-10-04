import csv
import pytest
from atomic_features.targets import build_targets, split_masks
from atomic_features.hierarchy import WIDTHS, compute_regrets, summarize_regrets


def test_aggregation_pools_category_model_trajectories():
    rows = []
    for model, category, value in [
        ("gemini", "a", 1.0),
        ("nemotron", "a", 0.0),
        ("nemotron", "b", 0.0),
    ]:
        for width in WIDTHS:
            rows.append(
                dict(
                    dataset="wordfreq",
                    model=model,
                    category_id=category,
                    level="one_letter_prefix",
                    width=str(width),
                    test_f1=str(value),
                )
            )
    stats = summarize_regrets(compute_regrets(rows), tolerance=0.02)
    assert stats[("one_letter_prefix", 512)].mean_test_f1 == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="Duplicate"):
        compute_regrets(rows + [rows[0]])
    with pytest.raises(ValueError, match="Incomplete"):
        compute_regrets(rows[1:])


def test_gbif_common_names_union_species_memberships_and_species_eligibility(tmp_path):
    (tmp_path / "categories_min100_species.tsv").write_text(
        "category_id\tlevel\tcategory\tn_species\nfamily:F1\tfamily\tF1\t100\nfamily:F2\tfamily\tF2\t101\nfamily:F3\tfamily\tF3\t99\n"
    )
    (tmp_path / "species_common_name_pairs.csv").write_text(
        "common_row_idx,scientificName,family\n7,Species A,F1\n7,Species B,F2\n8,Species C,F3\n"
    )
    rows = [
        dict(row_idx="7", nameType="common", text="shared name"),
        dict(row_idx="900", nameType="scientific", text="Species A"),
        dict(row_idx="8", nameType="common", text="rare"),
    ]
    targets = build_targets("gbif", rows, 100, tmp_path)
    assert [(t["category_id"], list(t["positive_rows"])) for t in targets] == [
        ("family:F1", [0, 1]),
        ("family:F2", [0]),
    ]
    assert [t["positive_count"] for t in targets] == [100, 101]


def test_prefix_rows_and_split_assignment_are_retained():
    rows = [
        dict(text="ab", split="test"),
        dict(text="ac", split="train"),
        dict(text="bb", split="test"),
    ]
    targets = build_targets("wordfreq", rows, min_count=2)
    assert len(targets) == 1 and list(targets[0]["positive_rows"]) == [0, 1]
    train, test = split_masks(rows)
    assert list(train) == [False, True, False] and list(test) == [True, False, True]
