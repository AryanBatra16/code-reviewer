"""
baselines.py - TF-IDF + LogisticRegression and RandomForest baselines.
Vuln classification only (before=1 vs after=0).
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MASKED_LABEL = -100


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_split_data(
    samples_df: pd.DataFrame,
    split: dict,
    split_name: str = "P1",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter samples_df by commit_ids in split."""
    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])
    train_df = samples_df[samples_df["commit_id"].isin(train_ids)]
    val_df = samples_df[samples_df["commit_id"].isin(val_ids)]
    test_df = samples_df[samples_df["commit_id"].isin(test_ids)]
    return train_df, val_df, test_df


def compute_pairwise_accuracy(
    test_df: pd.DataFrame, y_pred_proba: np.ndarray
) -> float:
    """
    Pairwise accuracy: fraction of test pairs where P(vuln|before) > P(vuln|after).
    """
    test_df = test_df.copy()
    test_df["pred_proba"] = y_pred_proba
    pair_groups = test_df.groupby("pair_id")
    correct = 0
    total = 0
    for pid, group in pair_groups:
        before = group[group["sample_type"] == "before"]
        after = group[group["sample_type"] == "after"]
        if len(before) == 1 and len(after) == 1:
            if before["pred_proba"].values[0] > after["pred_proba"].values[0]:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def run_baseline(
    name: str,
    model,
    vectorizer: TfidfVectorizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protocol: str,
    results_dir: Path,
) -> dict:
    """Train and evaluate one baseline model."""
    X_train = vectorizer.fit_transform(train_df["text"].astype(str).fillna(""))
    y_train = train_df["vuln"].values

    X_val = vectorizer.transform(val_df["text"].astype(str).fillna(""))
    X_test = vectorizer.transform(test_df["text"].astype(str).fillna(""))
    y_test = test_df["vuln"].values

    logger.info("Training %s on %d samples...", name, len(y_train))
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    logger.info("Training done in %.1fs", train_time)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pair_acc = compute_pairwise_accuracy(test_df, y_proba)

    result = {
        "model": name,
        "protocol": protocol,
        "seed": 42,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pair_accuracy": pair_acc,
        "train_time_sec": train_time,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    logger.info(
        "%s [%s]: Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f PairAcc=%.4f",
        name, protocol, acc, prec, rec, f1, pair_acc,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF baselines")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(".")
    processed_dir = root / cfg["data"]["processed_dir"]
    splits_dir = root / cfg["results"]["splits_dir"]
    results_dir = root / cfg["results"]["dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_quick" if args.quick else ""
    samples_path = processed_dir / f"samples{suffix}.csv"
    if not samples_path.exists():
        raise FileNotFoundError(f"Run data_prep.py first. Missing: {samples_path}")

    logger.info("Loading samples from %s ...", samples_path)
    samples_df = pd.read_csv(samples_path)

    all_results = []

    for protocol in ["P1", "P2", "P3"]:
        split_path = splits_dir / f"{protocol}{suffix}.json"
        if not split_path.exists():
            logger.warning("Split %s not found, skipping.", split_path)
            continue
        with open(split_path) as f:
            split = json.load(f)

        train_df, val_df, test_df = load_split_data(samples_df, split, protocol)

        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning("Protocol %s has empty train or test set, skipping.", protocol)
            continue

        # TF-IDF vectorizer
        vectorizer_b1 = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=200_000,
            strip_accents="unicode",
        )
        vectorizer_b2 = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=200_000,
            strip_accents="unicode",
        )

        # B1: TF-IDF + LR
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs")
        r1 = run_baseline("B1_TFIDF_LR", lr, vectorizer_b1, train_df, val_df, test_df, protocol, results_dir)
        all_results.append(r1)

        # B2: TF-IDF + RF
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
        r2 = run_baseline("B2_TFIDF_RF", rf, vectorizer_b2, train_df, val_df, test_df, protocol, results_dir)
        all_results.append(r2)

    # Save results
    results_df = pd.DataFrame(all_results)
    out_path = results_dir / "baselines_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Saved baseline results to %s", out_path)
    print("\nBaseline Results:")
    print(results_df[["model", "protocol", "accuracy", "precision", "recall", "f1", "pair_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
