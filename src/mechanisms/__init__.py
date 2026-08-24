from .scoring import continuation_margin
from .patching import exact_layer_patching_profile, top_k_layers, multi_layer_patch_recovery
from .attribution import attribution_patching_profile
from .signature import compute_failure_excess_signature, aggregate_stable_baseline
from .similarity import cosine_sim, spearman_sim, topk_jaccard

__all__ = [
    "continuation_margin",
    "exact_layer_patching_profile",
    "top_k_layers",
    "multi_layer_patch_recovery",
    "attribution_patching_profile",
    "compute_failure_excess_signature",
    "aggregate_stable_baseline",
    "cosine_sim",
    "spearman_sim",
    "topk_jaccard",
]
