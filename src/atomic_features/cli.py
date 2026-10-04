"""Command line interface; no repository paths or network dependencies."""

import argparse
import json
from pathlib import Path
import sys
from . import __version__


def main(argv=None):
    p = argparse.ArgumentParser(prog="atomic-features", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("verify", help="Verify all bundle members against the manifest")
    q.add_argument("bundle", type=Path)
    q = sub.add_parser("unpack", help="Safely extract and verify a release archive")
    q.add_argument("archive", type=Path)
    q.add_argument("output", type=Path)
    q = sub.add_parser(
        "reproduce", help="Recompute main empirical results from saved inputs"
    )
    q.add_argument("bundle", type=Path)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--candidates", type=Path)
    q = sub.add_parser(
        "encode", help="Encode a dense NPY embedding array using a released SAE"
    )
    q.add_argument("--models", type=Path, required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--embeddings", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--batch-size", type=int, default=256)
    q.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    q = sub.add_parser(
        "probe", help="Fit train-only probes and evaluate existing held-out splits"
    )
    q.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset directory inside the evaluation bundle",
    )
    q.add_argument("--activations", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--category", help="Optional exact category_id")
    q.add_argument("--min-tp", type=int, default=1)
    args = p.parse_args(argv)
    try:
        if args.command == "verify":
            from .bundles import verify

            m = verify(args.bundle)
            result = dict(
                status="passed",
                kind=m["kind"],
                files=len(m["files"]),
                bytes=sum(f["bytes"] for f in m["files"]),
            )
        elif args.command == "unpack":
            from .bundles import unpack

            m = unpack(args.archive, args.output)
            result = dict(status="passed", kind=m["kind"], output=str(args.output))
        elif args.command == "reproduce":
            from .reproduce import reproduce

            result = reproduce(args.bundle, args.output, args.candidates)
            result = {
                k: v for k, v in result.items() if k != "persistence_certificates"
            }
        elif args.command == "encode":
            import numpy as np
            from scipy.sparse import save_npz
            from .bundles import verify, safe_path
            from .sae import SAE

            if verify(args.models)["kind"] != "models":
                raise ValueError("Expected a models bundle")
            if args.output.exists():
                raise FileExistsError(args.output)
            model = SAE(safe_path(args.models, args.model))
            acts = model.encode(
                np.load(args.embeddings, mmap_mode="r", allow_pickle=False),
                args.batch_size,
                args.backend,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as f:
                save_npz(f, acts)
            result = dict(
                status="passed",
                rows=acts.shape[0],
                features=acts.shape[1],
                nonzero=acts.nnz,
            )
        elif args.command == "probe":
            result = probe(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        p.exit(1, f"atomic-features: {exc}\n")
    print(json.dumps(result, indent=2))


def probe(args):
    import numpy as np
    from scipy.sparse import load_npz
    from .bundles import write_json
    from .targets import read_rows, split_masks
    from .probes import fit_probe, evaluate_probe

    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_rows(args.dataset / "rows.csv")
    train, test = split_masks(rows)
    x = load_npz(args.activations).tocsr()
    labels = load_npz(args.dataset / "targets.npz").tocsc()
    targets = json.loads((args.dataset / "targets.json").read_text())
    if x.shape[0] != len(rows) or labels.shape != (len(rows), len(targets)):
        raise ValueError("Rows, activations and targets are not aligned")
    results = []
    for i, target in enumerate(targets):
        if args.category and target["category_id"] != args.category:
            continue
        y = labels[:, i].toarray().ravel().astype(bool)
        record = {"category_id": target["category_id"], "level": target["level"]}
        if not y[train].any() or not y[test].any():
            results.append(
                {**record, "status": "ineligible: no positive examples in a split"}
            )
            continue
        try:
            selected = fit_probe(x[train], y[train], args.min_tp)
        except ValueError as exc:
            if str(exc) != "No feature meets the training-positive requirement":
                raise
            results.append({**record, "status": str(exc)})
            continue
        results.append(
            {
                **record,
                "status": "fitted",
                "feature": selected.feature,
                "threshold": selected.threshold,
                "train": evaluate_probe(selected, x[train], y[train]),
                "test": evaluate_probe(selected, x[test], y[test]),
            }
        )
    if not results:
        raise ValueError("No matching categories")
    write_json(args.output, results)
    return dict(status="passed", categories=len(results), output=str(args.output))


if __name__ == "__main__":
    main()
