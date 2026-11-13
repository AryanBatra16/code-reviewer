"""
data_prep.py - Data preparation for VulnReview.

Parses Big-Vul CSV, extracts C/C++ patch hunks, builds before/after pairs,
creates CWE/severity labels, performs commit-grouped splits (P1/P2/P3).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh"}
FILE_DELIMITER = "<_**next**_>"
TOP_K_CWE = 10
MASKED_LABEL = -100

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

# ---------------------------------------------------------------------------
# Step 1: Parse files_changed
# ---------------------------------------------------------------------------

def parse_files_changed(raw: str) -> list[dict]:
    """Split on delimiter and json.loads each part; skip malformed."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = raw.split(FILE_DELIMITER)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            obj = json.loads(part)
            result.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass
    return result


def is_c_cpp(filename: str) -> bool:
    """Return True if filename has a C/C++ extension."""
    return Path(filename).suffix.lower() in C_EXTENSIONS


def split_into_hunks(patch: str) -> list[tuple[str, list[str]]]:
    """
    Split a unified-diff patch string into hunks.
    Returns list of (hunk_header, hunk_lines).
    hunk_header starts with '@@'.
    """
    hunks = []
    current_header = None
    current_lines: list[str] = []
    for line in patch.splitlines():
        if line.startswith("@@"):
            if current_header is not None:
                hunks.append((current_header, current_lines))
            current_header = line
            current_lines = []
        else:
            if current_header is not None:
                current_lines.append(line)
    if current_header is not None:
        hunks.append((current_header, current_lines))
    return hunks


def hunk_has_both(lines: list[str]) -> bool:
    """Return True if hunk has at least one removed line AND one added line."""
    has_minus = any(l.startswith("-") for l in lines)
    has_plus = any(l.startswith("+") for l in lines)
    return has_minus and has_plus


def build_pair_texts(hunk_lines: list[str]) -> tuple[str, str, list[int]]:
    """
    Build before/after texts from hunk lines.
    Returns:
        before: context + removed lines (marker stripped)
        after:  context + added lines (marker stripped)
        removed_line_indices: 0-based indices in `before` lines that were removed
    """
    before_lines = []
    after_lines = []
    removed_indices = []  # indices in before_lines that were '-' lines

    for line in hunk_lines:
        if line.startswith("-"):
            removed_indices.append(len(before_lines))
            before_lines.append(line[1:])  # strip '-'
        elif line.startswith("+"):
            after_lines.append(line[1:])   # strip '+'
        else:
            # context line: appears in both
            before_lines.append(line)
            after_lines.append(line)

    return "\n".join(before_lines), "\n".join(after_lines), removed_indices

# ---------------------------------------------------------------------------
# Step 2: Labels
# ---------------------------------------------------------------------------

def build_cwe_map(df: pd.DataFrame) -> tuple[dict[str, int], list[str]]:
    """
    Build CWE label map: top-10 CWEs get classes 0..9, 'Other'=10.
    Returns (cwe_to_idx, ordered_cwe_list).
    """
    cwe_counts = Counter()
    for cwe in df["cwe_id"].dropna():
        cwe = str(cwe).strip()
        if cwe and cwe.lower() not in ("", "nan", "none"):
            cwe_counts[cwe] += 1

    top_cwes = [cwe for cwe, _ in cwe_counts.most_common(TOP_K_CWE)]
    cwe_to_idx = {cwe: i for i, cwe in enumerate(top_cwes)}
    cwe_to_idx["Other"] = len(top_cwes)  # index 10
    ordered = top_cwes + ["Other"]
    logger.info("Top-10 CWEs: %s", top_cwes)
    return cwe_to_idx, ordered


def get_cwe_label(cwe_id: Any, cwe_to_idx: dict[str, int]) -> int:
    """Return CWE label index or MASKED_LABEL if missing."""
    if pd.isna(cwe_id) or str(cwe_id).strip().lower() in ("", "nan", "none"):
        return MASKED_LABEL
    cwe = str(cwe_id).strip()
    if cwe in cwe_to_idx:
        return cwe_to_idx[cwe]
    return cwe_to_idx["Other"]


def get_severity_label(score: Any) -> int:
    """
    0=low (<4.0), 1=medium (4.0-6.9), 2=high (>=7.0).
    Returns MASKED_LABEL if missing.
    """
    try:
        s = float(score)
        if s < 4.0:
            return 0
        elif s < 7.0:
            return 1
        else:
            return 2
    except (ValueError, TypeError):
        return MASKED_LABEL


SEVERITY_NAMES = ["low", "medium", "high"]

# ---------------------------------------------------------------------------
# Step 3: Main extraction
# ---------------------------------------------------------------------------

def extract_pairs(df: pd.DataFrame, cwe_to_idx: dict[str, int], quick: bool = False) -> list[dict]:
    """
    For each row, parse files_changed, extract C/C++ hunks,
    build before/after pairs with labels.
    Returns list of sample dicts.
    """
    samples = []
    pair_id = 0
    stats = {
        "total_rows": 0,
        "rows_with_files": 0,
        "c_cpp_files": 0,
        "total_hunks": 0,
        "hunks_with_both": 0,
        "commits_with_pairs": set(),
    }

    iter_df = df.head(500) if quick else df
    for _, row in tqdm(iter_df.iterrows(), total=len(iter_df), desc="Extracting pairs"):
        stats["total_rows"] += 1
        raw_files = row.get("files_changed", "")
        file_objects = parse_files_changed(raw_files)
        if not file_objects:
            continue
        stats["rows_with_files"] += 1

        commit_id = str(row.get("commit_id", "")).strip()
        cve_id = str(row.get("cve_id", "")).strip()
        cwe_id = row.get("cwe_id", None)
        score = row.get("score", None)
        publish_date = str(row.get("publish_date", "")).strip()
        project = str(row.get("project", "")).strip()

        row_has_pair = False
        for file_obj in file_objects:
            filename = file_obj.get("filename", "")
            patch = file_obj.get("patch", "")
            if not is_c_cpp(filename) or not patch:
                continue
            stats["c_cpp_files"] += 1

            hunks = split_into_hunks(patch)
            stats["total_hunks"] += len(hunks)

            for hunk_header, hunk_lines in hunks:
                if not hunk_has_both(hunk_lines):
                    continue
                stats["hunks_with_both"] += 1

                before_text, after_text, removed_indices = build_pair_texts(hunk_lines)

                # Drop pairs where before == after
                if before_text.strip() == after_text.strip():
                    continue

                # Build the two samples
                before_sample = {
                    "pair_id": pair_id,
                    "sample_type": "before",
                    "text": before_text,
                    "vuln": 1,
                    "cwe_label": get_cwe_label(cwe_id, cwe_to_idx),
                    "sev_label": get_severity_label(score),
                    "removed_line_indices": removed_indices,
                    "commit_id": commit_id,
                    "cve_id": cve_id,
                    "cwe_id": str(cwe_id) if not pd.isna(cwe_id) else "",
                    "score": float(score) if score is not None and not pd.isna(score) else None,
                    "publish_date": publish_date,
                    "project": project,
                    "filename": filename,
                    "hunk_header": hunk_header,
                }
                after_sample = {
                    "pair_id": pair_id,
                    "sample_type": "after",
                    "text": after_text,
                    "vuln": 0,
                    "cwe_label": MASKED_LABEL,  # always masked for after
                    "sev_label": MASKED_LABEL,
                    "removed_line_indices": [],
                    "commit_id": commit_id,
                    "cve_id": cve_id,
                    "cwe_id": "",
                    "score": float(score) if score is not None and not pd.isna(score) else None,
                    "publish_date": publish_date,
                    "project": project,
                    "filename": filename,
                    "hunk_header": hunk_header,
                }
                samples.append(before_sample)
                samples.append(after_sample)
                pair_id += 1
                row_has_pair = True

        if row_has_pair:
            stats["commits_with_pairs"].add(commit_id)

    stats["commits_with_pairs"] = len(stats["commits_with_pairs"])
    return samples, stats

# ---------------------------------------------------------------------------
# Step 4: Splits
# ---------------------------------------------------------------------------

def get_commit_to_date(df: pd.DataFrame) -> dict[str, str]:
    """Map commit_id -> publish_date."""
    return dict(zip(df["commit_id"].astype(str), df["publish_date"].astype(str)))


def get_commit_to_project(df: pd.DataFrame) -> dict[str, str]:
    """Map commit_id -> project."""
    return dict(zip(df["commit_id"].astype(str), df["project"].astype(str)))


def get_commit_to_cwe(samples: list[dict]) -> dict[str, str]:
    """Map commit_id -> cwe_id (from before samples)."""
    result = {}
    for s in samples:
        if s["sample_type"] == "before":
            result[s["commit_id"]] = s.get("cwe_id", "")
    return result


def group_by_commit(samples: list[dict]) -> dict[str, list[dict]]:
    """Group samples by commit_id."""
    groups = defaultdict(list)
    for s in samples:
        groups[s["commit_id"]].append(s)
    return groups


def split_p1_random(
    commit_ids: list[str],
    commit_to_cwe: dict[str, str],
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """Commit-grouped 80/10/10 random split, stratified by CWE."""
    rng = random.Random(seed)
    # Group commits by CWE
    cwe_to_commits: dict[str, list[str]] = defaultdict(list)
    for cid in commit_ids:
        cwe = commit_to_cwe.get(cid, "Unknown")
        cwe_to_commits[cwe].append(cid)

    train_ids, val_ids, test_ids = [], [], []
    for cwe, cids in cwe_to_commits.items():
        shuffled = cids[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0
        train_ids.extend(shuffled[:n_train])
        val_ids.extend(shuffled[n_train:n_train + n_val])
        test_ids.extend(shuffled[n_train + n_val:])

    return train_ids, val_ids, test_ids


def split_p2_temporal(
    commit_ids: list[str],
    commit_to_date: dict[str, str],
    seed: int,
    cutoff_year: int = 2017,
    test_years: set | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Temporal split: <=cutoff_year -> train+val, test_years -> test."""
    if test_years is None:
        test_years = {2018, 2019}

    train_val_ids = []
    test_ids = []
    skipped = 0
    for cid in commit_ids:
        date_str = commit_to_date.get(cid, "")
        if not date_str or date_str.lower() in ("nan", "none", ""):
            skipped += 1
            continue
        try:
            year = int(str(date_str)[:4])
        except (ValueError, IndexError):
            skipped += 1
            continue
        if year <= cutoff_year:
            train_val_ids.append(cid)
        elif year in test_years:
            test_ids.append(cid)
        # else: skip (outside range)

    logger.info("P2: %d train+val commits, %d test commits, %d skipped (missing date)",
                len(train_val_ids), len(test_ids), skipped)

    # 90/10 split of train_val, grouped by commit
    rng = random.Random(seed)
    rng.shuffle(train_val_ids)
    n_train = int(len(train_val_ids) * 0.9)
    train_ids = train_val_ids[:n_train]
    val_ids = train_val_ids[n_train:]
    return train_ids, val_ids, test_ids


def split_p3_cross_project(
    commit_ids: list[str],
    commit_to_project: dict[str, str],
    seed: int,
    train_projects: set | None = None,
    val_ratio: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """Cross-project: Chrome/Linux_kernel/Android -> train+val, rest -> test."""
    if train_projects is None:
        train_projects = {"Chrome", "Linux_kernel", "Android"}

    # normalize project names
    def normalize(p: str) -> str:
        return p.strip().lower().replace(" ", "_")

    train_norm = {normalize(p) for p in train_projects}

    train_val_ids = []
    test_ids = []
    for cid in commit_ids:
        proj = commit_to_project.get(cid, "")
        if normalize(proj) in train_norm:
            train_val_ids.append(cid)
        else:
            test_ids.append(cid)

    rng = random.Random(seed)
    rng.shuffle(train_val_ids)
    n_val = max(1, int(len(train_val_ids) * val_ratio))
    train_ids = train_val_ids[:-n_val] if n_val < len(train_val_ids) else train_val_ids
    val_ids = train_val_ids[-n_val:] if n_val < len(train_val_ids) else []
    return train_ids, val_ids, test_ids


def leakage_check(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    name: str = "split",
) -> bool:
    """Check no commit_id appears in multiple splits."""
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set
    ok = True
    if tv:
        logger.error("[%s] LEAKAGE: train∩val = %d commits: %s", name, len(tv), list(tv)[:5])
        ok = False
    if tt:
        logger.error("[%s] LEAKAGE: train∩test = %d commits: %s", name, len(tt), list(tt)[:5])
        ok = False
    if vt:
        logger.error("[%s] LEAKAGE: val∩test = %d commits: %s", name, len(vt), list(vt)[:5])
        ok = False
    if ok:
        logger.info("[%s] No commit leakage detected. ✓", name)
    return ok

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VulnReview data preparation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--quick", action="store_true", help="Use tiny subset for pipeline testing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    root = Path(".")
    raw_csv = root / cfg["data"]["raw_csv"]
    processed_dir = root / cfg["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = root / cfg["results"]["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    models_dir = root / cfg["results"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CSV from %s ...", raw_csv)
    df = pd.read_csv(raw_csv)
    n_rows = len(df)
    logger.info("Loaded %d rows.", n_rows)

    if args.quick:
        logger.warning("QUICK MODE: using first 500 rows only.")
        df = df.head(500)

    # Build CWE map
    cwe_to_idx, ordered_cwes = build_cwe_map(df)

    # Extract pairs
    logger.info("Extracting patch pairs...")
    samples, stats = extract_pairs(df, cwe_to_idx, quick=args.quick)

    # Print sanity stats
    print("\n" + "=" * 60)
    print("SANITY STATS (Step 1)")
    print("=" * 60)
    print(f"  CSV rows loaded             : {n_rows}")
    print(f"  Rows with file objects      : {stats['rows_with_files']}")
    print(f"  C/C++ file patches          : {stats['c_cpp_files']}")
    print(f"  Total hunks                 : {stats['total_hunks']}")
    print(f"  Hunks with both -/+ lines   : {stats['hunks_with_both']}")
    print(f"  Commits with valid pairs    : {stats['commits_with_pairs']}")
    print(f"  Total samples (before+after): {len(samples)}")
    print(f"  Total pairs                 : {len(samples) // 2}")

    if not args.quick:
        expected = {
            "c_cpp_files": (10501, 0.02),
            "total_hunks": (27603, 0.02),
            "hunks_with_both": (17560, 0.02),
            "commits_with_pairs": (2904, 0.02),
        }
        all_ok = True
        for key, (expected_val, tol) in expected.items():
            actual = stats[key]
            diff = abs(actual - expected_val) / expected_val
            status = "✓" if diff <= tol else "⚠ EXCEEDS 2% TOLERANCE"
            print(f"  [{status}] {key}: expected ~{expected_val}, got {actual} (diff {diff:.1%})")
            if diff > tol:
                all_ok = False
        if not all_ok:
            logger.warning("Some stats differ by >2%% from expected. Review before training.")
    print("=" * 60 + "\n")

    # Save samples to CSV (without removed_line_indices for CSV compatibility)
    samples_df = pd.DataFrame([
        {k: v for k, v in s.items() if k != "removed_line_indices"}
        for s in samples
    ])
    samples_df["removed_line_indices"] = [
        json.dumps(s["removed_line_indices"]) for s in samples
    ]
    out_csv = processed_dir / ("samples_quick.csv" if args.quick else "samples.csv")
    samples_df.to_csv(out_csv, index=False)
    logger.info("Saved %d samples to %s", len(samples), out_csv)

    # Build splits
    all_commit_ids = list(samples_df["commit_id"].unique())
    commit_to_date = dict(zip(df["commit_id"].astype(str), df["publish_date"].astype(str)))
    commit_to_project = dict(zip(df["commit_id"].astype(str), df["project"].astype(str)))
    commit_to_cwe = get_commit_to_cwe(samples)

    # P1: Random
    p1_train, p1_val, p1_test = split_p1_random(all_commit_ids, commit_to_cwe, seed=args.seed)
    leakage_check(p1_train, p1_val, p1_test, "P1")
    logger.info("P1: train=%d, val=%d, test=%d commits", len(p1_train), len(p1_val), len(p1_test))

    # P2: Temporal
    p2_train, p2_val, p2_test = split_p2_temporal(all_commit_ids, commit_to_date, seed=args.seed)
    leakage_check(p2_train, p2_val, p2_test, "P2")
    logger.info("P2: train=%d, val=%d, test=%d commits", len(p2_train), len(p2_val), len(p2_test))

    # P3: Cross-project
    # Try multiple project name variants
    p3_train, p3_val, p3_test = split_p3_cross_project(all_commit_ids, commit_to_project, seed=args.seed)
    leakage_check(p3_train, p3_val, p3_test, "P3")
    logger.info("P3: train=%d, val=%d, test=%d commits", len(p3_train), len(p3_val), len(p3_test))
    # Print unique project names for verification
    unique_projects = sorted(samples_df["project"].unique())
    logger.info("Unique projects (%d): %s", len(unique_projects), unique_projects[:20])

    # Save splits
    splits_data = {
        "P1": {"train": p1_train, "val": p1_val, "test": p1_test},
        "P2": {"train": p2_train, "val": p2_val, "test": p2_test},
        "P3": {"train": p3_train, "val": p3_val, "test": p3_test},
    }
    suffix = "_quick" if args.quick else ""
    for protocol, split in splits_data.items():
        out_path = splits_dir / f"{protocol}{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(split, f, indent=2)
        logger.info("Saved split %s to %s", protocol, out_path)

    # Save label maps
    label_maps = {
        "cwe_to_idx": cwe_to_idx,
        "ordered_cwes": ordered_cwes,
        "severity_map": {"low": 0, "medium": 1, "high": 2},
        "severity_thresholds": {"low": [0, 4.0], "medium": [4.0, 7.0], "high": [7.0, 10.1]},
        "masked_label": MASKED_LABEL,
        "top_k_cwe": TOP_K_CWE,
        "stats": {
            "c_cpp_files": stats["c_cpp_files"],
            "total_hunks": stats["total_hunks"],
            "hunks_with_both": stats["hunks_with_both"],
            "commits_with_pairs": stats["commits_with_pairs"],
            "total_samples": len(samples),
        },
    }
    labels_path = models_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(label_maps, f, indent=2)
    logger.info("Saved label maps to %s", labels_path)

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
