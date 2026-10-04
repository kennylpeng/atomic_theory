"""Top-K SAE inference from portable, non-pickle NumPy tensors."""

from pathlib import Path
import json
import numpy as np
from scipy.sparse import csr_matrix, vstack


class SAE:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "model.json").read_text())
        self.W_enc, self.W_dec, self.b_dec = [
            np.load(directory / (n + ".npy"), mmap_mode="r", allow_pickle=False)
            for n in ("W_enc", "W_dec", "b_dec")
        ]
        if (
            self.W_enc.ndim != 2
            or self.W_dec.shape != self.W_enc.T.shape
            or self.b_dec.shape != (self.W_enc.shape[0],)
        ):
            raise ValueError("Inconsistent SAE tensor shapes")
        self.k = self.metadata["top_k"]
        if not isinstance(self.k, int) or not 0 < self.k <= self.W_enc.shape[1]:
            raise ValueError("Invalid top_k")
        if (
            self.metadata.get("input_unit_norm") is not False
            or self.metadata.get("model_type") != "topk"
        ):
            raise ValueError("Only the paper's unnormalized TopK SAE is supported")
        if any(a.dtype != np.float32 for a in (self.W_enc, self.W_dec, self.b_dec)):
            raise ValueError("The release inference format requires float32 weights")

    def encode(self, inputs, batch_size=256, backend="numpy"):
        """Return CSR activations, retaining original feature IDs.

        NumPy resolves exact positive ties by smaller feature ID; torch uses
        torch.topk, matching the training implementation's tie convention.
        BLAS/device roundoff can affect nearly tied activations.
        """
        if inputs.ndim != 2 or inputs.shape[1] != self.W_enc.shape[0] or batch_size < 1:
            raise ValueError("Invalid input shape or batch size")
        if backend not in {"numpy", "torch"}:
            raise ValueError("backend must be numpy or torch")
        batches = []
        if backend == "torch":
            import torch

            enc = torch.from_numpy(np.array(self.W_enc))
            bias = torch.from_numpy(np.array(self.b_dec))
        for start in range(0, len(inputs), batch_size):
            x = np.asarray(inputs[start : start + batch_size], dtype=np.float32)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite embedding")
            if backend == "torch":
                with torch.no_grad():
                    h = torch.relu((torch.from_numpy(x.copy()) - bias) @ enc)
                    val, ids = torch.topk(h, self.k, dim=-1)
                    val, ids = val.numpy(), ids.numpy()
            else:
                h = np.maximum((x - self.b_dec) @ self.W_enc, 0)
                # Stable sorting makes the positive-tie convention explicit.
                ids = np.argsort(-h, axis=1, kind="stable")[:, : self.k]
                val = np.take_along_axis(h, ids, axis=1)
            rows = np.repeat(np.arange(len(x)), self.k)
            acts = csr_matrix(
                (val.ravel(), (rows, ids.ravel())), shape=(len(x), self.W_enc.shape[1])
            )
            acts.eliminate_zeros()
            batches.append(acts)
        return (
            vstack(batches, format="csr")
            if batches
            else csr_matrix((0, self.W_enc.shape[1]), dtype=np.float32)
        )

    def reconstruct(self, activations):
        return activations @ self.W_dec + self.b_dec
