"""Semantic clustering and hierarchy construction (Section 18, 34, 35).

Purely embedding-based: no mechanistic / activation information enters this
module. HDBSCAN gives a natural cluster hierarchy via its single-linkage
tree, which we cut at three distance thresholds to produce coarse/mid/fine
levels matching the worked example in Section 18.
"""
from __future__ import annotations

import numpy as np


def cluster_descriptions(embeddings: np.ndarray, min_cluster_size: int = 5,
                          min_samples: int = 3, metric: str = "euclidean"):
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)
    return clusterer, labels


def _medoid_description(embeddings: np.ndarray, descriptions: list[str], indices: list[int]) -> str:
    if not indices:
        return "unlabeled"
    sub = embeddings[indices]
    centroid = sub.mean(axis=0)
    dists = np.linalg.norm(sub - centroid, axis=1)
    return descriptions[indices[int(np.argmin(dists))]]


def _flat_clusters_at_threshold(clusterer, cut_distance: float, min_cluster_size: int) -> np.ndarray:
    return clusterer.single_linkage_tree_.get_clusters(cut_distance, min_cluster_size)


def build_hierarchy(
    embeddings: np.ndarray,
    descriptions: list[str],
    clusterer,
    min_cluster_size: int = 5,
) -> dict:
    """Cuts the HDBSCAN single-linkage tree at three distances (fine -> mid
    -> coarse, i.e. increasing cut distance = fewer, larger clusters) and
    assigns each resulting cluster a medoid-description-derived name.

    Returns {"fine": {...}, "mid": {...}, "coarse": {...}} where each level
    maps example index -> {"cluster_id": int, "label": str}.
    """
    tree = clusterer.single_linkage_tree_.to_numpy()
    distances = tree[:, 2]
    d_lo, d_hi = np.percentile(distances, [40, 85])
    cut_points = {
        "fine": float(np.percentile(distances, 25)),
        "mid": float(d_lo + (d_hi - d_lo) * 0.5),
        "coarse": float(np.percentile(distances, 90)),
    }

    levels = {}
    for level_name, cut in cut_points.items():
        labels = _flat_clusters_at_threshold(clusterer, cut, min_cluster_size)
        cluster_ids = sorted(set(labels.tolist()) - {-1})
        names = {}
        for cid in cluster_ids:
            idxs = [i for i, l in enumerate(labels) if l == cid]
            names[cid] = _medoid_description(embeddings, descriptions, idxs)
        assignment = {}
        for i, lab in enumerate(labels):
            assignment[i] = {
                "cluster_id": int(lab),
                "label": names.get(int(lab), "noise") if lab != -1 else "noise",
            }
        levels[level_name] = {"assignment": assignment, "cut_distance": cut}
    return levels
