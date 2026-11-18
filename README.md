# Archetypal Analysis: Finding the Extremes That Explain Your Data

**CMPE 255 – Data Mining · Short Story Assignment · Fall 2026**
**Author:** Anita Agasaveeran

This repository holds my short story on **Archetypal Analysis (AA)**: a Medium article, a slide deck, and a recorded presentation. It is based on the survey paper *A Survey on Archetypal Analysis* plus supporting survey literature.

---

## Deliverables

| Deliverable | Link |
|---|---|
| Medium article | [Archetypal Analysis: Finding the Extremes That Explain Your Data](https://medium.com/@anita.agasaveeran/archetypal-analysis-finding-the-extremes-that-explain-your-data-ed56a53a62da) — source in [`article/medium_article.md`](article/medium_article.md) |
| Slide deck (SlideShare) | [Archetypal Analysis: Finding the Extremes That Explain Your Data](https://www.scribd.com/document/1096573656/Archetypal-Analysis-Finding-the-Extremes-That-Explain-Your-Data) |
| Slides (PDF, in this repo) | [`slides/slidedeck.pdf`](slides/slidedeck.pdf) |
| Presentation video (10–15 min) | [`video/presentation.mp4`](video/presentation.mp4) · also on [Loom](https://www.loom.com/share/1223b36313dc4c29a47c5de87e15b9bb) |
| Source paper (PDF) | [`2504.12392v3.pdf`](2504.12392v3.pdf) |

---

## The Paper

> A. Alcacer, I. Epifanio, S. Mair, and M. Mørup, **"A Survey on Archetypal Analysis,"** arXiv:2504.12392v3 [stat.ME], Oct. 2026.
> https://arxiv.org/abs/2504.12392

It is the first comprehensive survey of archetypal analysis. Cutler and Breiman introduced AA in 1994, and it now appears in hundreds of papers across biology, chemistry, climate science, computer science, and the social sciences.

### Why it fits CMPE 255

AA is an **unsupervised** method for **dimensionality reduction**, **matrix factorization**, and **pattern discovery**. It is closely related to k-means clustering, PCA, and NMF, all of which the course covers.

---

## The Short Story in a Nutshell

**The idea.** AA finds a small set of *archetypes*: extreme, "pure" profiles that lie on the boundary of the data's convex hull. It then describes every observation as a **convex combination** (a weighted mix whose weights sum to 1) of those archetypes. Running AA with three archetypes on handwritten 9s from MNIST gives "narrow and straight," "narrow and sloped," and "wide and straight." Every other 9 is a blend of those three.

**How it compares with familiar methods:**

| Method | What the components are | Where they sit |
|---|---|---|
| k-means | Typical representatives (centroids) | Interior of the data |
| PCA | Orthogonal directions of variance | Not data points at all |
| NMF | Nonnegative parts | Not guaranteed to lie in the data domain |
| **AA** | **Extreme representatives (archetypes)** | **Boundary of the convex hull** |

**What the article covers:**
1. **Intuition and history.** Cutler and Breiman's original formulation, and the growth of AA's citations since then.
2. **Why AA is useful.** It stays faithful to the observed data, its weights are easy to read, and it reveals extremes and trade-offs (Pareto-like structure).
3. **The AA family of variants.** Relaxed, kernel, probabilistic, functional, deep, and archetypoid analysis (where the archetypes are actual data points).
4. **Practical guidance.** Optimization algorithms and their cost, initialization (FurthestSum, AA++), scaling to large data (coresets, reduced-space AA), and choosing the number of archetypes *K*.
5. **Applications.** Evolutionary biology, NMR spectroscopy, hyperspectral remote sensing, congressional voting records, anomaly detection, and more.
6. **Limitations.** Nonconvex optimization, sensitivity to outliers, no standard way to choose *K*, scalability, and non-unique solutions.
7. **Open problems.** Archetypes that change over time, AA on manifolds, fair and differentially private AA, and data that lacks "pure" forms.
8. **My two cents.** My own assessment of the survey and of when AA beats clustering or NMF in practice.

---

## Repository Structure

```
.
├── README.md
├── 2504.12392v3.pdf          # Source survey paper
├── article/
│   └── medium_article.md     # Medium article draft (images referenced from ../images)
├── code/
│   ├── archetypal.py         # Small NumPy implementation of AA + init strategies + stability metrics
│   └── make_figures.py       # Regenerates every figure in images/
├── images/                   # All original figures used in the article and slides
├── requirements.txt
├── slides/
│   └── slidedeck.pdf         # 20-slide deck (also on SlideShare)
└── video/
    └── presentation.mp4      # Recorded presentation (also on Loom)
```

## Original Experiments

All figures are my own, generated with the code in [`code/`](code/):

| Figure | What it shows |
|---|---|
| `fig00_hero.png` | Points colored by their mix of three archetypes |
| `fig01_aa_vs_kmeans_pca_nmf.png` | AA vs. k-means vs. PCA vs. NMF on the same 2-D data |
| `fig02_varying_k.png` | Effect of the number of archetypes K (and the K > M+1 degeneracy) |
| `fig03_digits_components.png` | Components learned on handwritten 2s by each method |
| `fig04_simplex_plot.png` | Simplex plot: every digit as a mix of three archetypal 2s |
| `fig05_model_selection.png` | Choosing K: variance explained + NMI / archetype-similarity stability |
| `fig06_init_strategies.png` | Uniform vs. FurthestSum vs. AA++ initial picks |
| `fig07_init_ablation.png` | Initialization ablation on the digits dataset (12 seeds per method) |
| `fig08_outlier_sensitivity.png` | One outlier hijacking an archetype |
| `fig09_aa_family_tree.png` | Taxonomy of AA extensions |
| `fig10_method_comparison_table.png` | Comparison table (Medium can't render tables) |

### Reproduce the figures

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python code/make_figures.py          # all figures (~5 min on a laptop)
python code/make_figures.py hero toy # or a subset
```

Only scikit-learn's bundled digits dataset is used, so there's nothing to download.

---

## References

**Archetypal analysis**
- A. Cutler and L. Breiman, "Archetypal Analysis," *Technometrics* 36(4), 1994.
- M. Mørup and L. K. Hansen, "Archetypal analysis for machine learning and data mining," *Neurocomputing* 80, 2012.
- S. Mair and U. Brefeld, "Coresets for archetypal analysis," *NeurIPS* 2019.
- S. Mair and J. Sjölund, "Archetypal analysis++: Rethinking the initialization strategy," *TMLR* 2024.
- T. Fel et al., "Archetypal SAE: Adaptive and stable dictionary learning for concept extraction in large vision models," *ICML* 2025.

**Related surveys**
- A. K. Jain, "Data clustering: 50 years beyond K-means," *Pattern Recognition Letters* 31(8), 2010.
- Y.-X. Wang and Y.-J. Zhang, "Nonnegative Matrix Factorization: A Comprehensive Review," *IEEE TKDE* 25(6), 2013.
- N. Gillis, "The Why and How of Nonnegative Matrix Factorization," arXiv:1401.5226, 2014.
- J. M. Bioucas-Dias et al., "Hyperspectral unmixing overview," *IEEE JSTARS* 5(2), 2012.

The full reference list is at the end of the article.

---

## Disclaimer

The Medium article and slides are my own work, written from scratch for this assignment. Figures taken from the source paper are credited to the original authors.
