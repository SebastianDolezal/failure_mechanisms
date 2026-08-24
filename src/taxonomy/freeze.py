"""Freezing and loading the semantic taxonomy (Section 18, 40).

Once written, taxonomy_v1.json is immutable for all confirmatory analyses.
Its sha256 is recorded alongside it so downstream scripts can assert they
are reading the exact frozen version (Section 18: "commit its hash/version
to the repository").
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..datasets.schema import _abs


def freeze_taxonomy(hierarchy: dict, descriptions: list[str], base_ids: list[str],
                     embeddings: np.ndarray, out_path: str, version: str = "v1") -> dict:
    payload = {
        "version": version,
        "levels": {},
        "n_examples": len(descriptions),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
    }
    # Cluster centroids per level, so novel (e.g. SVAMP) descriptions can be
    # mapped on without recomputing the clustering.
    for level_name, level in hierarchy.items():
        clusters: dict[str, dict] = {}
        for idx, info in level["assignment"].items():
            cid = str(info["cluster_id"])
            clusters.setdefault(cid, {"label": info["label"], "member_indices": [], "member_base_ids": []})
            clusters[cid]["member_indices"].append(idx)
            clusters[cid]["member_base_ids"].append(base_ids[idx])
        for cid, c in clusters.items():
            idxs = c["member_indices"]
            c["centroid"] = embeddings[idxs].mean(axis=0).tolist() if idxs else None
            c["n_members"] = len(idxs)
        payload["levels"][level_name] = {
            "cut_distance": level["cut_distance"],
            "clusters": clusters,
        }

    out_path_abs = _abs(out_path)
    out_path_abs.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path_abs, "w") as f:
        json.dump(payload, f, indent=2)

    sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with open(str(out_path_abs) + ".sha256", "w") as f:
        f.write(sha + "\n")

    return {"path": str(out_path_abs), "sha256": sha}


def load_taxonomy(path: str) -> dict:
    with open(_abs(path), "r") as f:
        return json.load(f)


def map_description_to_taxonomy(
    embedding: np.ndarray,
    taxonomy: dict,
    level: str = "mid",
    novelty_threshold: float = 0.35,
) -> dict:
    """Nearest-centroid mapping used for the frozen SVAMP replication
    (Section 40). If the closest centroid's cosine distance exceeds
    `novelty_threshold`, the description is labeled "novel/unmapped" rather
    than force-fit, per spec ("allow novel/unmapped")."""
    clusters = taxonomy["levels"][level]["clusters"]
    best_cid, best_sim = None, -1.0
    for cid, c in clusters.items():
        if c["centroid"] is None:
            continue
        centroid = np.array(c["centroid"])
        sim = float(embedding @ centroid / (np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8))
        if sim > best_sim:
            best_sim, best_cid = sim, cid
    if best_cid is None or (1 - best_sim) > novelty_threshold:
        return {"cluster_id": None, "label": "novel/unmapped", "similarity": best_sim}
    return {"cluster_id": best_cid, "label": clusters[best_cid]["label"], "similarity": best_sim}
