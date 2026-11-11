"""Generate every original figure used in the Medium article and slides.

Usage (from the repo root):
    python code/make_figures.py            # all figures
    python code/make_figures.py hero toy   # a subset, by name
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import NMF, PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archetypal import (archetype_similarity, fit_aa, init_indices, nmi,  # noqa: E402
                        variance_explained)

OUT = Path(__file__).resolve().parent.parent / "images"
OUT.mkdir(exist_ok=True)

COL = dict(data="#9aa0a6", aa="#1b9e77", km="#d95f02", pca="#3778c2",
           nmf="#7570b3", hull="#5f6368", out="#d62728", ink="#202124")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5f6368", "axes.labelcolor": COL["ink"],
    "xtick.color": "#5f6368", "ytick.color": "#5f6368",
})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def toy_triangle(n=400, seed=0):
    rng = np.random.default_rng(seed)
    V = np.array([[0.0, 0.0], [4.0, 0.5], [1.5, 3.5]])
    W = rng.dirichlet([0.6, 0.6, 0.6], n)
    return W @ V + rng.normal(0, 0.12, (n, 2)) + 0.6  # shift keeps data > 0 for NMF


def digits(cls=None):
    d = load_digits()
    X = d.data / 16.0
    return X if cls is None else X[d.target == cls]


def draw_hull(ax, X, **kw):
    h = ConvexHull(X)
    ax.add_patch(Polygon(X[h.vertices], closed=True, fill=False,
                         ec=COL["hull"], lw=1, ls="--", **kw))


def draw_polytope(ax, A, color=COL["aa"], alpha=0.18):
    order = ConvexHull(A).vertices if len(A) > 2 else np.arange(len(A))
    ax.add_patch(Polygon(A[order], closed=True, fc=color, ec=color, lw=2, alpha=alpha))
    ax.add_patch(Polygon(A[order], closed=True, fill=False, ec=color, lw=2))


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


def digit_img(v, ax, cmap="gray_r", vmin=0, vmax=1):
    ax.imshow(v.reshape(8, 8), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_hero():
    X = toy_triangle(600, seed=3)
    r = fit_aa(X, 3)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    draw_hull(ax, X)
    draw_polytope(ax, r["A"], alpha=0.12)
    w = r["S"]
    rgb = w @ np.array([[0.11, 0.62, 0.47], [0.85, 0.37, 0.01], [0.46, 0.44, 0.70]])
    ax.scatter(*X.T, c=np.clip(rgb, 0, 1), s=14, alpha=0.85, lw=0)
    ax.scatter(*r["A"].T, s=260, marker="*", c=COL["ink"], zorder=5)
    for k, a in enumerate(r["A"]):
        ax.annotate(f"archetype {k + 1}", a, xytext=(10, 8), textcoords="offset points",
                    fontsize=12, weight="bold", color=COL["ink"])
    ax.set_title("Every point is a mix of a few extremes", fontsize=16, loc="left")
    clean(ax)
    ax.set_xlim(X[:, 0].min() - 0.4, X[:, 0].max() + 1.6)
    save(fig, "fig00_hero.png")


def fig_compare_methods():
    X = toy_triangle()
    aa = fit_aa(X, 3)
    km = KMeans(3, n_init=10, random_state=0).fit(X)
    pca = PCA(2).fit(X)
    nmf = NMF(2, init="nndsvda", max_iter=2000, random_state=0).fit(X)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    titles = ["Archetypal Analysis (K=3)", "k-means (K=3)", "PCA (2 components)", "NMF (2 components)"]
    lim = (X.min(0) - 0.5, X.max(0) + 0.5)
    for ax, t in zip(axes, titles):
        ax.scatter(*X.T, s=8, c=COL["data"], alpha=0.6, lw=0)
        draw_hull(ax, X)
        ax.set_title(t)
        clean(ax)
        ax.set_xlim(-0.2, lim[1][0]); ax.set_ylim(-0.2, lim[1][1])

    draw_polytope(axes[0], aa["A"])
    axes[0].scatter(*aa["A"].T, s=180, marker="*", c=COL["aa"], ec="k", zorder=5)
    axes[0].text(0.02, 0.02, "extremes on the hull", transform=axes[0].transAxes, color=COL["aa"])

    axes[1].scatter(*km.cluster_centers_.T, s=160, marker="X", c=COL["km"], ec="k", zorder=5)
    axes[1].text(0.02, 0.02, "averages inside the data", transform=axes[1].transAxes, color=COL["km"])

    mu = pca.mean_
    for i, (v, ev) in enumerate(zip(pca.components_, pca.explained_variance_)):
        d = v * 2.2 * np.sqrt(ev)
        axes[2].annotate("", mu + d, mu - d, arrowprops=dict(arrowstyle="<->", color=COL["pca"],
                                                             lw=2.5 if i == 0 else 1.5))
    axes[2].text(0.02, 0.02, "orthogonal directions", transform=axes[2].transAxes, color=COL["pca"])

    for h in nmf.components_:
        end = h / np.linalg.norm(h) * 6
        axes[3].plot([0, end[0]], [0, end[1]], c=COL["nmf"], lw=2.5)
    axes[3].scatter([0], [0], c=COL["nmf"], s=30)
    axes[3].text(0.45, 0.02, "a cone from the origin", transform=axes[3].transAxes, color=COL["nmf"])
    fig.suptitle("Same data, four views: what each method calls a 'component'", fontsize=15, y=0.98)
    save(fig, "fig01_aa_vs_kmeans_pca_nmf.png")


def fig_varying_k():
    X = toy_triangle()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, K in zip(axes, [2, 3, 4, 5]):
        r = fit_aa(X, K)
        ax.scatter(*X.T, s=8, c=COL["data"], alpha=0.6, lw=0)
        draw_hull(ax, X)
        if K == 2:
            ax.plot(*r["A"].T, c=COL["aa"], lw=3)
        else:
            draw_polytope(ax, r["A"])
        ax.scatter(*r["A"].T, s=160, marker="*", c=COL["aa"], ec="k", zorder=5)
        ax.set_title(f"K = {K}\nreconstruction error (RSS) = {r['rss']:.1f}", fontsize=12)
        clean(ax)
    fig.suptitle("More archetypes -> the green polytope grows toward the data's convex hull (dashed)",
                 fontsize=14, y=1.06)
    fig.text(0.5, -0.04, "Note K=4 and K=5 in 2-D: once K > M+1 the extra archetypes crowd the same corners "
             "- a degenerate case the survey warns about.", ha="center", color="#5f6368")
    save(fig, "fig02_varying_k.png")


def fig_digits_components(cls=2, K=4):
    X = digits(cls)
    aa = fit_aa(X, K, seed=1)
    km = KMeans(K, n_init=10, random_state=0).fit(X)
    pca = PCA(K).fit(X)
    nmf = NMF(K, init="nndsvda", max_iter=3000, random_state=0).fit(X)

    rows = [("Archetypal\nAnalysis", aa["A"], "gray_r", 0, 1),
            ("k-means\ncentroids", km.cluster_centers_, "gray_r", 0, 1),
            ("PCA\ncomponents", pca.components_, "coolwarm", -0.35, 0.35),
            ("NMF\ncomponents", nmf.components_ / nmf.components_.max(1, keepdims=True), "gray_r", 0, 1)]
    fig, axes = plt.subplots(4, K, figsize=(1.8 * K + 1.6, 8))
    for i, (name, comps, cmap, lo, hi) in enumerate(rows):
        for k in range(K):
            digit_img(comps[k], axes[i, k], cmap, lo, hi)
        axes[i, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=12, weight="bold",
                              labelpad=12)
    fig.suptitle(f"Handwritten '{cls}'s (sklearn digits, n={len(X)}): what K={K} components look like",
                 fontsize=13)
    fig.text(0.5, 0.01, "AA returns extreme but real-looking styles; k-means returns blurry averages;\n"
             "PCA/NMF return parts or contrasts that are not digits at all.",
             ha="center", fontsize=11, color="#5f6368")
    save(fig, "fig03_digits_components.png")


def fig_simplex(cls=2):
    X = digits(cls)
    r = fit_aa(X, 3, seed=2)
    S, A = r["S"], r["A"]
    V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    P = S @ V

    fig, ax = plt.subplots(figsize=(8, 7.4))
    ax.add_patch(Polygon(V, closed=True, fc="#e8f5ee", ec=COL["aa"], lw=2))
    dom = S.argmax(1)
    cols = np.array([COL["aa"], COL["km"], COL["nmf"]])
    ax.scatter(*P.T, s=22, c=cols[dom], alpha=0.75, lw=0)
    offsets = [(-0.12, -0.10), (0.12, -0.10), (0, 0.13)]
    for k in range(3):
        ab = AnnotationBbox(OffsetImage(A[k].reshape(8, 8), zoom=7, cmap="gray_r"),
                            V[k] + offsets[k], frameon=True,
                            bboxprops=dict(ec=cols[k], lw=3))
        ax.add_artist(ab)
    # Three example digits with their mixing weights
    picks = [int(np.argmin(np.abs(S - t).sum(1))) for t in
             ([0.34, 0.33, 0.33], [0.6, 0.4, 0.0], [0.05, 0.45, 0.5])]
    boxes = [np.array([-0.2, 0.62]), np.array([0.5, -0.24]), np.array([1.2, 0.62])]
    for n, box in zip(picks, boxes):
        ab = AnnotationBbox(OffsetImage(X[n].reshape(8, 8), zoom=4, cmap="gray_r"), P[n],
                            xybox=box, xycoords="data", boxcoords="data", frameon=True,
                            bboxprops=dict(ec="#5f6368", lw=1),
                            arrowprops=dict(arrowstyle="-", color="#5f6368"))
        ax.add_artist(ab)
        w = " / ".join(f"{x:.0%}" for x in S[n])
        ax.text(box[0], box[1] + 0.105, w, ha="center", fontsize=9.5, weight="bold")
        ax.scatter(*P[n], s=60, c="k", zorder=6)
    ax.set_xlim(-0.42, 1.42); ax.set_ylim(-0.45, 1.18)
    clean(ax)
    ax.set_title("Each '2' lives inside a triangle spanned by three archetypal '2's", fontsize=13, pad=14)
    ax.text(0.5, -0.43, "Position = mixing weights (they sum to 100%). Corners = archetypes.\n"
            "Labels give each example's weights on the green / orange / purple archetype.",
            ha="center", color="#5f6368", fontsize=10)
    save(fig, "fig04_simplex_plot.png")


def fig_model_selection(Ks=range(2, 11), restarts=5):
    X = digits()
    tot = np.sum((X - X.mean(0)) ** 2)
    ve_aa, nmi_aa, sim_aa, ve_km, ve_pca = [], [], [], [], []
    pca = PCA(max(Ks)).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    for K in Ks:
        fits = [fit_aa(X, K, seed=s, n_iter=200) for s in range(restarts)]
        ves = [variance_explained(X, f["rss"]) for f in fits]
        ve_aa.append((np.mean(ves), np.std(ves)))
        pairs = list(combinations(fits, 2))
        nmi_aa.append(np.mean([nmi(a["S"], b["S"]) for a, b in pairs]))
        sim_aa.append(np.mean([archetype_similarity(a["A"], b["A"], X) for a, b in pairs]))
        ve_km.append(1 - KMeans(K, n_init=5, random_state=0).fit(X).inertia_ / tot)
        ve_pca.append(cum[K - 1])
        print(f"  K={K}: VE={ve_aa[-1][0]:.3f}  NMI={nmi_aa[-1]:.3f}  sim={sim_aa[-1]:.3f}")
    Ks = list(Ks)
    m, s = np.array(ve_aa).T

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.plot(Ks, ve_pca, "o-", c=COL["pca"], label="PCA (upper bound, unconstrained)")
    a1.errorbar(Ks, m, yerr=s, fmt="o-", c=COL["aa"], capsize=3, lw=2.5, label=f"AA (mean ± sd, {restarts} restarts)")
    a1.plot(Ks, ve_km, "o-", c=COL["km"], label="k-means (hard assignment)")
    a1.set_xlabel("number of components K"); a1.set_ylabel("variance explained")
    a1.set_title("Fit: AA sits between k-means and PCA"); a1.legend(frameon=False, fontsize=9)
    a1.grid(alpha=0.25)

    a2.plot(Ks, nmi_aa, "o-", c=COL["aa"], lw=2.5, label="NMI(S, S')  - weights agree?")
    a2.plot(Ks, sim_aa, "s--", c=COL["ink"], label="sim(A, A')  - archetypes agree?")
    a2.set_ylim(0, 1.02); a2.set_xlabel("number of archetypes K"); a2.set_ylabel("agreement across restarts")
    a2.set_title("Stability: do restarts find the same solution?"); a2.legend(frameon=False, fontsize=9)
    a2.grid(alpha=0.25)
    fig.suptitle("Choosing K on the full digits data (N=1797, M=64): fit vs. reproducibility", fontsize=14, y=1.03)
    save(fig, "fig05_model_selection.png")


def fig_init_toy():
    X = toy_triangle(seed=5)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    names = {"uniform": "Uniform (random points)", "furthestsum": "FurthestSum", "aa++": "AA++"}
    for ax, (m, title) in zip(axes, names.items()):
        idx = init_indices(X, 4, m, np.random.default_rng(7))
        ax.scatter(*X.T, s=8, c=COL["data"], alpha=0.6, lw=0)
        draw_hull(ax, X)
        draw_polytope(ax, X[idx], color=COL["km"], alpha=0.12)
        ax.scatter(*X[idx].T, s=150, marker="s", c=COL["km"], ec="k", zorder=5)
        for j, p in enumerate(X[idx]):
            ax.annotate(str(j + 1), p, xytext=(6, 6), textcoords="offset points", weight="bold")
        ax.set_title(title); clean(ax)
    fig.suptitle("Initial archetypes (K=4) picked by three strategies - numbers show pick order",
                 fontsize=14, y=1.03)
    save(fig, "fig06_init_strategies.png")


def fig_init_ablation(K=8, seeds=12):
    X = digits()
    methods = [("uniform", "Uniform"), ("furthestsum", "FurthestSum"), ("aa++", "AA++")]
    cols = [COL["data"], COL["pca"], COL["aa"]]
    res = {m: [fit_aa(X, K, init=m, seed=s, n_iter=150) for s in range(seeds)] for m, _ in methods}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for (m, name), c in zip(methods, cols):
        H = np.array([variance_explained(X, f["history"][:120]) for f in res[m]
                      if len(f["history"]) >= 120])
        if len(H):
            med = np.median(H, 0)
            a1.plot(med, c=c, lw=2.5, label=name)
            a1.fill_between(np.arange(H.shape[1]), H.min(0), H.max(0), color=c, alpha=0.15)
    a1.set_xlabel("iteration"); a1.set_ylabel("variance explained")
    a1.set_title("Convergence (median, min-max band)"); a1.legend(frameon=False); a1.grid(alpha=0.25)
    a1.set_ylim(0.2, None)

    finals = [[variance_explained(X, f["rss"]) for f in res[m]] for m, _ in methods]
    bp = a2.boxplot(finals, patch_artist=True, widths=0.5)
    a2.set_xticks([1, 2, 3], [n for _, n in methods])
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    for i, f in enumerate(finals, 1):
        a2.scatter(np.full(len(f), i) + np.random.default_rng(i).normal(0, 0.04, len(f)), f, s=14, c="k")
    a2.set_ylabel("final variance explained"); a2.set_title(f"Final fit after 150 iterations ({seeds} seeds)")
    a2.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Ablation: does initialisation matter? (digits, K={K})", fontsize=14, y=1.03)
    for (m, name), f in zip(methods, finals):
        print(f"  {name}: mean={np.mean(f):.4f} sd={np.std(f):.4f}")
    save(fig, "fig07_init_ablation.png")


def fig_outlier():
    X = toy_triangle(seed=1)
    o = np.array([12.0, 9.0])
    Xo = np.vstack([X, o])
    clean_fit, dirty_fit = fit_aa(X, 3), fit_aa(Xo, 3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, D, r, t in [(axes[0], X, clean_fit, "400 clean points"),
                        (axes[1], Xo, dirty_fit, "Same 400 points + ONE outlier")]:
        ax.scatter(*D.T, s=8, c=COL["data"], alpha=0.6, lw=0)
        draw_polytope(ax, r["A"])
        ax.scatter(*r["A"].T, s=180, marker="*", c=COL["aa"], ec="k", zorder=5)
        ax.set_title(t); clean(ax); ax.set_xlim(0, 13); ax.set_ylim(0, 9.8)
    A0 = clean_fit["A"]
    axes[1].add_patch(Polygon(A0, closed=True, fill=False, ec=COL["aa"], lw=1.5, ls=":"))
    axes[1].scatter(*o, s=120, c=COL["out"], zorder=6)
    axes[1].annotate("1 outlier out of 401 points\ndrags an archetype with it", o, xytext=(4.0, 8.6),
                     color=COL["out"], arrowprops=dict(arrowstyle="->", color=COL["out"]))
    axes[1].text(0.2, 0.2, "dotted = archetypes without the outlier", color=COL["aa"], fontsize=10)
    fig.suptitle("Limitation: archetypes live on the boundary, so outliers pull them hard", fontsize=14, y=1.02)
    save(fig, "fig08_outlier_sensitivity.png")


def _box(ax, xy, w, h, text, fc, fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="none"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight,
            color=COL["ink"], wrap=True)


def fig_taxonomy():
    groups = [
        ("Where archetypes live", "#d7efe4", ["Relaxed AA\n(may leave the hull)", "Archetypoids (ADA)\n(must be real points)"]),
        ("Nonlinear spaces", "#dbe7f7", ["Kernel AA", "Deep AA / AAnet\n(learned latent space)", "Archetypal SAE\n(LLM/vision concepts)"]),
        ("Other data types", "#fde6d6", ["Probabilistic AA\n(binary, counts)", "Functional AA\n(curves)", "Ordinal / nominal /\ninterval AA"]),
        ("Robust & scalable", "#e7e5f4", ["Robust AA\n(Huber, Tukey)", "Weighted AA\n& coresets", "Reduced-space AA\n(PCA first)"]),
        ("Beyond rows", "#f6e3ea", ["Biarchetype AA\n(rows + columns)", "Multi-subject AA\n(e.g. fMRI)", "Sparse AA\n(hyperspectral)"]),
    ]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.5); ax.axis("off")
    _box(ax, (5.3, 5.4), 4.4, 0.85, "Archetypal Analysis\nCutler & Breiman, 1994", "#1b9e77", 13, "bold")
    ax.texts[-1].set_color("white")
    for i, (title, fc, items) in enumerate(groups):
        x = 0.25 + i * 2.95
        ax.plot([7.5, x + 1.3], [5.4, 4.65], c="#bdc1c6", lw=1.2, zorder=0)
        _box(ax, (x, 4.0), 2.6, 0.6, title, fc, 11, "bold")
        for j, it in enumerate(items):
            _box(ax, (x, 3.0 - j * 1.05), 2.6, 0.85, it, fc, 9.5)
    ax.set_title("The AA family tree (based on Table II of Alcacer et al.)", fontsize=15, loc="left")
    save(fig, "fig09_aa_family_tree.png")


def fig_comparison_table():
    rows = [
        ["", "k-means", "PCA", "NMF", "Archetypal Analysis"],
        ["Component is...", "an average\n(centroid)", "a direction\nof variance", "a nonnegative\n'part'", "an extreme\n(archetype)"],
        ["Lives...", "inside the data", "anywhere", "inside a cone,\nmaybe off-data", "on the convex\nhull boundary"],
        ["Each point gets...", "1 hard label", "real-valued\nscores", "nonnegative\nweights", "weights that\nsum to 1"],
        ["Easy to explain?", "yes", "not really", "sometimes", "yes - 'x% of\narchetype k'"],
        ["Main weakness", "misses extremes", "hard to read", "non-unique", "outliers, choosing K,\nnonconvex fit"],
    ]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis("off")
    tb = ax.table(cellText=rows, cellLoc="center", loc="center", colWidths=[0.17, 0.18, 0.18, 0.18, 0.22])
    tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 3.1)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor(COL["ink"] if c else "white"); cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif c == 0:
            cell.set_facecolor("#f1f3f4"); cell.get_text().set_weight("bold")
        elif c == 4:
            cell.set_facecolor("#d7efe4")
        else:
            cell.set_facecolor("#fafafa")
    rows[0][0] = ""
    ax.set_title("AA vs. the usual suspects", fontsize=15, loc="left")
    save(fig, "fig10_method_comparison_table.png")


FIGS = dict(hero=fig_hero, toy=fig_compare_methods, k=fig_varying_k, digits=fig_digits_components,
            simplex=fig_simplex, select=fig_model_selection, init=fig_init_toy,
            ablation=fig_init_ablation, outlier=fig_outlier, tree=fig_taxonomy, table=fig_comparison_table)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        print(f"[{name}]")
        FIGS[name]()
