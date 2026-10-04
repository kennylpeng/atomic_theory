"""Small loading helpers for the dictionaries used in full experiments."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_dictionary(
    path: Path,
    kind: str,
    width: int | None = None,
    *,
    sae_direction: str = "decoder",
) -> torch.Tensor:
    """Load an SAE, KMeans, or PCA dictionary and normalize its rows."""
    if kind == "sae":
        if sae_direction not in {"decoder", "encoder"}:
            raise ValueError(f"Unknown SAE direction: {sae_direction}")
        # mmap avoids eagerly copying a large checkpoint when supported.
        try:
            checkpoint = torch.load(_paper_location(path), map_location="cpu", mmap=True, weights_only=False)
        except TypeError:  # Older PyTorch versions do not support mmap.
            checkpoint = torch.load(_paper_location(path), map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        if sae_direction == "decoder":
            dictionary = state["W_dec"].detach().float().cpu()
        else:
            # W_enc is stored as (input dimension, number of features), so
            # transpose it to keep one feature direction per dictionary row.
            dictionary = state["W_enc"].detach().T.float().cpu()
    else:
        # KMeans centroids and PCA components are NumPy arrays.
        array = np.load(_paper_location(path), mmap_mode="r")
        if kind == "pca":
            if width is None:
                raise ValueError("width is required when loading PCA directions")
            # The first m components form a width-m PCA dictionary.
            array = array[:width]
        elif kind != "kmeans":
            raise ValueError(f"Unknown dictionary kind: {kind}")
        # Copy out of the read-only memory map before tensor conversion.
        dictionary = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))

    if dictionary.ndim != 2:
        raise ValueError(f"Expected a matrix in {path}, got shape {tuple(dictionary.shape)}")
    # Normalized row dot products are cosine similarities.
    return F.normalize(dictionary, dim=1)
