"""Primary figures (Section 47). Static figures use matplotlib; Figure 6
(splits/merges) uses plotly for an interactive Sankey diagram.
"""
from __future__ import annotations

import numpy as np


def _save(fig, out_path: str):
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def _fit_line(ax, x: np.ndarray, y: np.ndarray, color: str = "crimson"):
    """Overlays a degree-1 least-squares trend line, purely as a visual aid
    (never a substitute for the actual fitted statistical models in
    src/statistics/). Skipped when x has ~zero variance: a constant column
    makes np.polyfit's design matrix rank-deficient, which can raise
    LinAlgError ("SVD did not converge") on some LAPACK backends - since this
    is only a plot annotation, that's a reason to omit the line, not to
    crash the run that already computed the real statistics."""
    if len(x) < 2 or np.ptp(x) < 1e-10:
        return
    z = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=color, lw=2)


def fig1_experimental_design(out_path: str):
    """Schematic of the semantic / causal / correspondence pipeline
    (Figure 1)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")

    boxes = {
        "problem": (0.5, 0.95, "Benchmark problem"),
        "variants": (0.5, 0.83, "Meaning-preserving variants"),
        "pair": (0.5, 0.71, "Clean <-> failed pair"),
        "sem": (0.22, 0.55, "Semantic path:\nphenomenological description -> taxonomy"),
        "mech": (0.78, 0.55, "Causal path:\nraw patching profile -> stable-pair subtraction -> D_i"),
        "test": (0.5, 0.35, "Correspondence test\nT_i <-> D_i"),
    }
    for key, (x, y, label) in boxes.items():
        ax.add_patch(mpatches.FancyBboxPatch((x - 0.19, y - 0.045), 0.38, 0.09,
                                              boxstyle="round,pad=0.02", fc="#eef2f7", ec="#334155"))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    arrows = [("problem", "variants"), ("variants", "pair"), ("pair", "sem"), ("pair", "mech"),
              ("sem", "test"), ("mech", "test")]
    for a, b in arrows:
        x0, y0, _ = boxes[a]
        x1, y1, _ = boxes[b]
        ax.annotate("", xy=(x1, y1 + 0.045), xytext=(x0, y0 - 0.045),
                    arrowprops=dict(arrowstyle="->", color="#334155"))

    ax.set_title("Figure 1. Experimental design")
    _save(fig, out_path)
    plt.close(fig)


def fig2_reliability_hierarchy(
    same_method_sims: np.ndarray,
    diff_failure_same_category_sims: np.ndarray,
    matched_diff_category_sims: np.ndarray,
    out_path: str,
):
    """Distributions for the reliability ordering in Section 2 / 26:
    same failure across methods > different failures, same semantic type >
    matched different semantic types."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [same_method_sims, diff_failure_same_category_sims, matched_diff_category_sims]
    labels = ["Same failure\n(different methods)", "Different failures\n(same category)",
              "Matched different\ncategories"]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Mechanistic similarity")
    ax.set_title("Figure 2. Reliability hierarchy")
    _save(fig, out_path)
    plt.close(fig)


def fig3_continuous_correspondence(S_sem: np.ndarray, C_mech: np.ndarray, out_path: str, beta1: float | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(S_sem, C_mech, alpha=0.35, s=14)
    _fit_line(ax, S_sem, C_mech)
    ax.set_xlabel("Semantic description similarity  S_ij^sem")
    ax.set_ylabel("Failure-excess causal similarity  C_ij")
    title = "Figure 3. Continuous semantic-mechanistic correspondence"
    if beta1 is not None:
        title += f"  (beta1={beta1:.3f})"
    ax.set_title(title)
    _save(fig, out_path)
    plt.close(fig)


def fig4_primary_category_result(same_cat: np.ndarray, diff_cat: np.ndarray, delta: float,
                                   ci: tuple[float, float], out_path: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].violinplot([same_cat, diff_cat], showmeans=True)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Same category", "Matched different category"])
    axes[0].set_ylabel("Mechanistic similarity C_ij")

    axes[1].bar(["Delta"], [delta], yerr=[[delta - ci[0]], [ci[1] - delta]], capsize=8, color="#2563eb")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("Delta = S_same - S_diff")
    fig.suptitle("Figure 4. Primary category result (H2/H3)")
    _save(fig, out_path)
    plt.close(fig)


def fig5_resolution_curve(levels: list[str], delta_values: list[float], delta_ci: list[tuple[float, float]],
                            out_path: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(levels))
    yerr = np.array([[d - lo, hi - d] for d, (lo, hi) in zip(delta_values, delta_ci)]).T
    ax.errorbar(x, delta_values, yerr=yerr, fmt="o-", capsize=5, color="#0891b2")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.set_ylabel("Mechanistic coherence (Delta)")
    ax.set_xlabel("Semantic granularity")
    ax.set_title("Figure 5. Semantic-resolution curve")
    _save(fig, out_path)
    plt.close(fig)


def fig6_splits_merges_sankey(semantic_labels: list[str], mechanistic_labels: list[str], out_path: str):
    """Sankey diagram of semantic categories <-> independently-discovered
    mechanistic groups (Section 36). Saved as an interactive HTML file."""
    import plotly.graph_objects as go
    import pandas as pd

    df = pd.DataFrame({"sem": semantic_labels, "mech": [f"M:{m}" for m in mechanistic_labels]})
    counts = df.groupby(["sem", "mech"]).size().reset_index(name="n")

    sem_nodes = sorted(df["sem"].unique())
    mech_nodes = sorted(df["mech"].unique())
    all_nodes = sem_nodes + mech_nodes
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=16, label=all_nodes),
        link=dict(
            source=[node_idx[s] for s in counts["sem"]],
            target=[node_idx[m] for m in counts["mech"]],
            value=counts["n"].tolist(),
        ),
    )])
    fig.update_layout(title_text="Figure 6. Semantic categories <-> mechanistic groups (splits/merges)")
    fig.write_html(out_path)


def fig7_intervention_transfer(C_ij: np.ndarray, R_transfer: np.ndarray, same_category: np.ndarray, out_path: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(same_category, "#2563eb", "#9ca3af")
    ax.scatter(C_ij, R_transfer, c=colors, alpha=0.5, s=14)
    _fit_line(ax, C_ij, R_transfer)
    ax.set_xlabel("Mechanistic similarity C_ij")
    ax.set_ylabel("Intervention transfer recovery R_{A->B}")
    ax.set_title("Figure 7. Intervention transfer vs. mechanistic similarity")
    _save(fig, out_path)
    plt.close(fig)
