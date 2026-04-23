"""
Visualizations for Claim Prediction Market experiment.

Generates four publication-quality charts from output/results.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

PALETTE = {
    "Skeptic": "#e74c3c",
    "Optimist": "#2ecc71",
    "Methodologist": "#3498db",
    "DomainExpert": "#9b59b6",
    "BaseRate": "#f39c12",
}

OUTPUT_DIR = Path(__file__).parent / "output"


def load_data() -> dict:
    with open(OUTPUT_DIR / "results.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Chart 1: Predictor Brier Scores
# ---------------------------------------------------------------------------

def chart_brier_scores(data: dict) -> None:
    predictors = data["predictor_brier_scores"]
    market_brier = data["market_ensemble_brier"]

    # Sort by score (best first)
    sorted_names = sorted(predictors.keys(), key=lambda n: predictors[n]["mean_brier"])
    scores = [predictors[n]["mean_brier"] for n in sorted_names]
    colors = [PALETTE.get(n, "#95a5a6") for n in sorted_names]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(sorted_names, scores, color=colors, edgecolor="white",
                   linewidth=1.2, height=0.55, zorder=3)

    # Market ensemble line
    ax.axvline(market_brier, color="#2c3e50", linestyle="-", linewidth=2,
               label=f"Market Ensemble ({market_brier:.3f})", zorder=4)

    # Chance baseline
    ax.axvline(0.25, color="#e74c3c", linestyle="--", linewidth=1.8,
               label="Chance Baseline (0.250)", zorder=4)

    # Value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(score + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=10, fontweight="bold",
                color="#2c3e50")

    ax.set_xlabel("Mean Brier Score (lower = better)")
    ax.set_title("Predictor Accuracy: Brier Scores Compared")
    ax.set_xlim(0, 0.35)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.invert_yaxis()

    fig.savefig(OUTPUT_DIR / "brier_scores.png")
    plt.close(fig)
    print("  Saved brier_scores.png")


# ---------------------------------------------------------------------------
# Chart 2: Market Consensus vs Actual Outcome
# ---------------------------------------------------------------------------

def chart_calibration(data: dict) -> None:
    claims = data["per_claim"]

    mean_preds = [c["mean_prediction"] for c in claims]
    actuals = [1.0 if c["ground_truth"] else 0.0 for c in claims]
    is_surprising = [c["is_surprising"] for c in claims]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Jitter actuals slightly for visibility
    rng = np.random.default_rng(42)
    jittered_actuals = [a + rng.uniform(-0.04, 0.04) for a in actuals]

    # Plot claims as scatter
    for i, (mp, ja, surp) in enumerate(zip(mean_preds, jittered_actuals, is_surprising)):
        color = "#e74c3c" if surp else ("#2ecc71" if ja > 0.5 else "#3498db")
        marker = "D" if surp else "o"
        size = 120 if surp else 70
        ax.scatter(mp, ja, c=color, marker=marker, s=size, zorder=5,
                   edgecolors="white", linewidths=0.8)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.5,
            label="Perfect Calibration")

    # Calibration curve from buckets
    cal = data["calibration"]
    bucket_x = []
    bucket_y = []
    bucket_n = []
    for bucket_key, stats in sorted(cal.items()):
        bucket_x.append(stats["mean_predicted"])
        bucket_y.append(stats["actual_verified_rate"])
        bucket_n.append(stats["n"])

    ax.plot(bucket_x, bucket_y, "o-", color="#e67e22", linewidth=2.5,
            markersize=10, markeredgecolor="white", markeredgewidth=1.5,
            label="Calibration Curve", zorder=6)

    # Annotate bucket sizes
    for bx, by, bn in zip(bucket_x, bucket_y, bucket_n):
        ax.annotate(f"n={bn}", (bx, by), textcoords="offset points",
                    xytext=(8, -12), fontsize=9, color="#7f8c8d")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", label="Verified (expected)"),
        mpatches.Patch(facecolor="#3498db", label="Unverified"),
        mpatches.Patch(facecolor="#e74c3c", label="Surprising (novel)"),
        plt.Line2D([0], [0], color="#e67e22", marker="o", linewidth=2.5,
                   markersize=8, label="Calibration Curve"),
        plt.Line2D([0], [0], color="black", linestyle="--", alpha=0.4,
                   label="Perfect Calibration"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

    ax.set_xlabel("Market Consensus Probability (mean of 5 predictors)")
    ax.set_ylabel("Actual Outcome (1 = Verified, 0 = Unverified)")
    ax.set_title("Market Calibration: Predicted vs Actual")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Unverified", "Verified"])

    fig.savefig(OUTPUT_DIR / "calibration_scatter.png")
    plt.close(fig)
    print("  Saved calibration_scatter.png")


# ---------------------------------------------------------------------------
# Chart 3: Claim Novelty Map
# ---------------------------------------------------------------------------

def chart_novelty_map(data: dict) -> None:
    claims = data["per_claim"]

    fig, ax = plt.subplots(figsize=(10, 7))

    for c in claims:
        mp = c["mean_prediction"]
        conf = c["confidence"]
        # Evidence count = sum of for+against from predictions reasoning length proxy
        # Use number of predictions as a proxy for evidence, but we actually
        # want evidence_total. Let's estimate from the claim data.
        # Since we have 5 predictors for each, use the spread of predictions as bubble size
        preds = [p["probability"] for p in c["predictions"]]
        spread = max(preds) - min(preds)
        n_preds = len(c["predictions"])

        # Bubble size based on number of predictions (all 5, so use spread instead)
        size = max(100, spread * 800)

        surp = c["is_surprising"]
        gt = c["ground_truth"]

        if surp:
            color = "#e74c3c"
            alpha = 0.85
            edgecolor = "#c0392b"
            linewidth = 2.5
        elif gt:
            color = "#2ecc71"
            alpha = 0.5
            edgecolor = "#27ae60"
            linewidth = 1
        else:
            color = "#3498db"
            alpha = 0.5
            edgecolor = "#2980b9"
            linewidth = 1

        ax.scatter(mp, conf, s=size, c=color, alpha=alpha,
                   edgecolors=edgecolor, linewidths=linewidth, zorder=5)

    # Quadrant labels
    ax.axhline(0.65, color="#bdc3c7", linestyle=":", linewidth=1, zorder=2)
    ax.axvline(0.5, color="#bdc3c7", linestyle=":", linewidth=1, zorder=2)

    ax.text(0.15, 0.95, "SURPRISING\n(Novel Discoveries)", fontsize=10,
            fontweight="bold", color="#e74c3c", ha="center", va="top",
            transform=ax.transAxes, alpha=0.7)
    ax.text(0.85, 0.95, "EXPECTED\n(Trivial/Confirmed)", fontsize=10,
            fontweight="bold", color="#27ae60", ha="center", va="top",
            transform=ax.transAxes, alpha=0.7)
    ax.text(0.15, 0.05, "REJECTED\n(Predictably Wrong)", fontsize=10,
            fontweight="bold", color="#3498db", ha="center", va="bottom",
            transform=ax.transAxes, alpha=0.7)
    ax.text(0.85, 0.05, "OVERCONFIDENT\n(Wrong Despite Consensus)", fontsize=10,
            fontweight="bold", color="#95a5a6", ha="center", va="bottom",
            transform=ax.transAxes, alpha=0.7)

    # Annotate the surprising claim
    for c in claims:
        if c["is_surprising"]:
            ax.annotate(
                c["claim_id"].split("/")[-1],
                (c["mean_prediction"], c["confidence"]),
                textcoords="offset points", xytext=(-15, 12),
                fontsize=9, fontweight="bold", color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
            )

    legend_elements = [
        plt.scatter([], [], s=100, c="#e74c3c", edgecolors="#c0392b",
                    linewidths=2, label="Surprising (verified, market predicted failure)"),
        plt.scatter([], [], s=100, c="#2ecc71", edgecolors="#27ae60",
                    alpha=0.5, label="Verified (expected)"),
        plt.scatter([], [], s=100, c="#3498db", edgecolors="#2980b9",
                    alpha=0.5, label="Unverified"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", framealpha=0.9, fontsize=9)

    ax.set_xlabel("Market Probability (mean predictor consensus)")
    ax.set_ylabel("Author's Stated Confidence")
    ax.set_title("Claim Novelty Map: Consensus vs Confidence")
    ax.set_xlim(0.15, 0.95)
    ax.set_ylim(0.35, 1.05)

    fig.savefig(OUTPUT_DIR / "novelty_map.png")
    plt.close(fig)
    print("  Saved novelty_map.png")


# ---------------------------------------------------------------------------
# Chart 4: Predictor Agreement Heatmap
# ---------------------------------------------------------------------------

def chart_predictor_heatmap(data: dict) -> None:
    claims = data["per_claim"]
    predictor_names = ["Skeptic", "Optimist", "Methodologist", "DomainExpert", "BaseRate"]

    # Build matrix: rows=claims, cols=predictors
    n_claims = len(claims)
    matrix = np.zeros((n_claims, len(predictor_names)))
    claim_labels = []
    ground_truths = []

    for i, c in enumerate(claims):
        # Short label
        parts = c["claim_id"].split("/")
        label = parts[-1]  # e.g., "claim-001"
        if len(parts) > 1:
            run = parts[0].split()[-1]  # e.g., date or "v2"
            label = f"{run}/{label}"
        claim_labels.append(label)
        ground_truths.append(c["ground_truth"])

        pred_map = {p["predictor"]: p["probability"] for p in c["predictions"]}
        for j, pname in enumerate(predictor_names):
            matrix[i, j] = pred_map.get(pname, 0.5)

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(predictor_names)))
    ax.set_xticklabels(predictor_names, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(n_claims))

    # Color y-labels by ground truth
    ax.set_yticklabels(claim_labels, fontsize=9)
    for i, (label, gt) in enumerate(zip(claim_labels, ground_truths)):
        color = "#27ae60" if gt else "#e74c3c"
        ax.get_yticklabels()[i].set_color(color)

    # Add text annotations
    for i in range(n_claims):
        for j in range(len(predictor_names)):
            val = matrix[i, j]
            text_color = "white" if val < 0.3 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    # Mark surprising claims
    for i, c in enumerate(claims):
        if c["is_surprising"]:
            rect = plt.Rectangle((-0.5, i - 0.5), len(predictor_names), 1,
                                 fill=False, edgecolor="#e74c3c",
                                 linewidth=2.5, linestyle="-")
            ax.add_patch(rect)

    # Ground truth column
    gt_ax = fig.add_axes([0.92, ax.get_position().y0, 0.025,
                          ax.get_position().y1 - ax.get_position().y0])
    gt_colors = np.array([[1, 0] if gt else [0, 1] for gt in ground_truths])
    gt_matrix = np.array([[1.0] if gt else [0.0] for gt in ground_truths])
    gt_ax.imshow(gt_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    gt_ax.set_xticks([0])
    gt_ax.set_xticklabels(["Truth"], fontsize=8, rotation=35, ha="right")
    gt_ax.set_yticks([])
    for i, gt in enumerate(ground_truths):
        gt_ax.text(0, i, "V" if gt else "X", ha="center", va="center",
                   fontsize=8, fontweight="bold",
                   color="white" if gt else "white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label("Predicted P(Verified)")

    ax.set_title("Predictor Agreement Heatmap\n(green claim labels = verified, red = unverified)")

    fig.savefig(OUTPUT_DIR / "predictor_heatmap.png")
    plt.close(fig)
    print("  Saved predictor_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating visualizations...")
    data = load_data()
    chart_brier_scores(data)
    chart_calibration(data)
    chart_novelty_map(data)
    chart_predictor_heatmap(data)
    print("Done.")


if __name__ == "__main__":
    main()
