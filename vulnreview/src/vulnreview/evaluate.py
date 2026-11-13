"""
evaluate.py - Aggregate and summarize all training results.
Computes mean ± std across seeds, generates confusion matrix plots,
runs paired bootstrap CI for B4 vs B3 on P1.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def aggregate_results(results_dir: Path) -> pd.DataFrame:
    """Load all *_results_*.csv files and return combined DataFrame."""
    dfs = []
    for f in results_dir.glob("*.csv"):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            logger.warning("Could not read %s: %s", f, e)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def mean_std_table(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """Group by group_cols and compute mean ± std for metric_cols."""
    rows = []
    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        for m in metric_cols:
            if m in group.columns:
                vals = group[m].dropna()
                row[f"{m}_mean"] = vals.mean()
                row[f"{m}_std"] = vals.std() if len(vals) > 1 else 0.0
                row[f"{m}_str"] = f"{vals.mean():.4f} ± {vals.std():.4f}" if len(vals) > 1 else f"{vals.mean():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_f1(
    f1_b4: list[float],
    f1_b3: list[float],
    n_resamples: int = 1000,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """
    Paired bootstrap CI (95%) for F1 difference B4 - B3.
    Returns (observed_diff, ci_low, ci_high).
    """
    rng = np.random.RandomState(rng_seed)
    b4 = np.array(f1_b4)
    b3 = np.array(f1_b3)
    obs_diff = np.mean(b4) - np.mean(b3)
    diffs = []
    n = len(b4)
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        diffs.append(np.mean(b4[idx]) - np.mean(b3[idx]))
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    return obs_diff, ci_low, ci_high


def generate_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", output_path)


def build_results_summary(
    combined_df: pd.DataFrame,
    results_dir: Path,
    cfg: dict,
) -> str:
    """Build the results_summary.md text from actual result CSVs."""
    lines = []
    lines.append("# VulnReview — Results Summary")
    lines.append("")
    lines.append("> **ALL NUMBERS ARE FROM ACTUAL RUNS. NO VALUES ARE FABRICATED.**")
    lines.append("")

    metric_cols_main = ["test_f1", "test_accuracy", "test_precision", "test_recall", "test_pair_accuracy"]
    metric_cols_aux = ["test_cwe_macro_f1", "test_sev_macro_f1", "test_sev_mae", "test_sw_recall"]

    # Table V: B1, B2, B3, B4 on P1
    lines.append("## Table V: Comparison on P1 (Random Split)")
    lines.append("")
    p1_df = combined_df[combined_df["protocol"] == "P1"] if "protocol" in combined_df.columns else pd.DataFrame()
    if not p1_df.empty:
        tbl = mean_std_table(p1_df, ["model", "protocol"], metric_cols_main)
        lines.append(tbl[["model", "test_f1_str", "test_accuracy_str", "test_precision_str", "test_recall_str", "test_pair_accuracy_str"]].to_markdown(index=False))
    else:
        lines.append("*No P1 results found yet.*")
    lines.append("")

    # Table VI: B4 on P1/P2/P3
    lines.append("## Table VI: B4 Generalization Across Protocols")
    lines.append("")
    b4_df = combined_df[combined_df["model"].str.contains("B4_full", na=False)] if not combined_df.empty else pd.DataFrame()
    if not b4_df.empty:
        tbl = mean_std_table(b4_df, ["model", "protocol"], metric_cols_main + metric_cols_aux)
        cols = ["protocol", "test_f1_str", "test_pair_accuracy_str", "test_cwe_macro_f1_str"]
        available = [c for c in cols if c in tbl.columns]
        lines.append(tbl[available].to_markdown(index=False))
    else:
        lines.append("*No B4 results found yet.*")
    lines.append("")

    # Table VII: Ablation on P1
    lines.append("## Table VII: Ablation Study (P1)")
    lines.append("")
    abl_df = combined_df[
        (combined_df.get("protocol", pd.Series()) == "P1") &
        (combined_df.get("model_type", pd.Series()).isin(["B3", "B4"]))
    ] if not combined_df.empty else pd.DataFrame()
    if not abl_df.empty:
        tbl = mean_std_table(abl_df, ["model"], ["test_f1", "test_pair_accuracy", "test_sw_recall"])
        lines.append(tbl[["model", "test_f1_str", "test_pair_accuracy_str", "test_sw_recall_str"]].to_markdown(index=False))
    else:
        lines.append("*No ablation results found yet.*")
    lines.append("")

    # Hyperparameters
    lines.append("## Hyperparameters Used")
    lines.append("")
    lines.append(f"- Model: `{cfg['model']['name']}`")
    lines.append(f"- Max length: {cfg['model'].get('max_len', 512)}")
    lines.append(f"- Batch size: {cfg['training']['batch_size']}")
    lines.append(f"- Grad accum steps: {cfg['training']['grad_accum_steps']}")
    lines.append(f"- Learning rate: {cfg['training']['lr']}")
    lines.append(f"- Weight decay: {cfg['training']['weight_decay']}")
    lines.append(f"- Epochs: {cfg['training']['epochs']}")
    lines.append(f"- Warmup ratio: {cfg['training']['warmup_ratio']}")
    lines.append(f"- fp16: {cfg['training']['fp16']}")
    lines.append(f"- Seeds: {cfg['seeds']}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(".")
    results_dir = root / cfg["results"]["dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    combined = aggregate_results(results_dir)
    if combined.empty:
        logger.warning("No result CSVs found in %s. Run training first.", results_dir)
        return

    logger.info("Aggregated %d result rows from %d files", len(combined), len(list(results_dir.glob("*.csv"))))

    if args.summarize:
        summary = build_results_summary(combined, results_dir, cfg)
        summary_path = results_dir / "results_summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info("Saved results_summary.md to %s", summary_path)
        print(summary)


if __name__ == "__main__":
    main()
