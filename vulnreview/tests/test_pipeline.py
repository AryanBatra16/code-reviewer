"""
tests/test_pipeline.py - pytest suite for VulnReview.

Tests:
1. test_parse_files_changed - JSON parsing and delimiter splitting
2. test_hunk_splitting - hunk parser
3. test_hunk_has_both - filter for -/+ lines
4. test_build_pair_texts - before/after pair builder
5. test_cwe_label_mapping - CWE label map
6. test_severity_label - severity bucketing
7. test_split_leakage_p1 - no commit overlap in P1 split
8. test_split_leakage_p2 - no commit overlap in P2 split
9. test_split_leakage_p3 - no commit overlap in P3 split
10. test_predict_smoke - predict.py loads and runs (requires model; skip if missing)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parse_files_changed_single():
    from vulnreview.data_prep import parse_files_changed
    raw = json.dumps({"filename": "foo.c", "patch": "@@ -1,2 +1,3 @@\n-old\n+new\n ctx"})
    result = parse_files_changed(raw)
    assert len(result) == 1
    assert result[0]["filename"] == "foo.c"


def test_parse_files_changed_multiple():
    from vulnreview.data_prep import parse_files_changed, FILE_DELIMITER
    obj1 = json.dumps({"filename": "a.c", "patch": "p1"})
    obj2 = json.dumps({"filename": "b.cpp", "patch": "p2"})
    raw = obj1 + FILE_DELIMITER + obj2
    result = parse_files_changed(raw)
    assert len(result) == 2
    assert result[0]["filename"] == "a.c"
    assert result[1]["filename"] == "b.cpp"


def test_parse_files_changed_malformed():
    from vulnreview.data_prep import parse_files_changed, FILE_DELIMITER
    good = json.dumps({"filename": "ok.c", "patch": "x"})
    bad = "NOT_JSON{{{broken"
    raw = good + FILE_DELIMITER + bad
    result = parse_files_changed(raw)
    assert len(result) == 1


def test_parse_files_changed_empty():
    from vulnreview.data_prep import parse_files_changed
    assert parse_files_changed("") == []
    assert parse_files_changed(None) == []


def test_hunk_splitting_basic():
    from vulnreview.data_prep import split_into_hunks
    patch = "@@ -1,3 +1,4 @@\n context\n-removed\n+added\n@@ -10,2 +11,3 @@\n ctx2\n-r2\n+a2"
    hunks = split_into_hunks(patch)
    assert len(hunks) == 2
    assert hunks[0][0].startswith("@@")
    assert any(l.startswith("-") for l in hunks[0][1])
    assert any(l.startswith("+") for l in hunks[0][1])


def test_hunk_has_both():
    from vulnreview.data_prep import hunk_has_both
    assert hunk_has_both([" ctx", "-removed", "+added"])
    assert not hunk_has_both([" ctx", "+added"])
    assert not hunk_has_both([" ctx", "-removed"])
    assert not hunk_has_both([" ctx"])


def test_build_pair_texts():
    from vulnreview.data_prep import build_pair_texts
    hunk_lines = [" context", "-vuln_line", "+fixed_line", " more_ctx"]
    before, after, removed = build_pair_texts(hunk_lines)
    assert "vuln_line" in before
    assert "fixed_line" not in before
    assert "fixed_line" in after
    assert "vuln_line" not in after
    assert "context" in before
    assert "context" in after
    assert 1 in removed


def test_cwe_label_mapping():
    from vulnreview.data_prep import get_cwe_label, build_cwe_map
    import pandas as pd
    # 12 unique CWEs so top 10 are selected and others go to Other
    cwes = [f"CWE-{i}" for i in range(12)] + ["CWE-0"] * 5
    df = pd.DataFrame({"cwe_id": cwes})
    cwe_to_idx, ordered = build_cwe_map(df)
    assert "Other" in cwe_to_idx
    assert "CWE-0" in cwe_to_idx
    assert cwe_to_idx["CWE-0"] == 0
    from vulnreview.data_prep import MASKED_LABEL
    assert get_cwe_label(None, cwe_to_idx) == MASKED_LABEL
    assert get_cwe_label("CWE-999-unseen", cwe_to_idx) == cwe_to_idx["Other"]
    assert get_cwe_label("CWE-0", cwe_to_idx) == cwe_to_idx["CWE-0"]


def test_severity_label():
    from vulnreview.data_prep import get_severity_label, MASKED_LABEL
    assert get_severity_label(2.5) == 0
    assert get_severity_label(5.0) == 1
    assert get_severity_label(7.0) == 2
    assert get_severity_label(10.0) == 2
    assert get_severity_label(None) == MASKED_LABEL
    assert get_severity_label("") == MASKED_LABEL
    assert get_severity_label("nan") == MASKED_LABEL


# ---------------------------------------------------------------------------
# Split leakage tests
# ---------------------------------------------------------------------------

def _load_split_if_exists(protocol: str) -> dict | None:
    for suffix in ["", "_quick"]:
        path = ROOT / "splits" / f"{protocol}{suffix}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


@pytest.mark.parametrize("protocol", ["P1", "P2", "P3"])
def test_split_no_leakage(protocol: str):
    """No commit_id should appear in more than one split partition."""
    split = _load_split_if_exists(protocol)
    if split is None:
        pytest.skip(f"Split {protocol} not found — run data_prep.py first")

    train_set = set(split["train"])
    val_set = set(split["val"])
    test_set = set(split["test"])

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    assert len(tv) == 0, f"[{protocol}] train∩val leakage: {list(tv)[:5]}"
    assert len(tt) == 0, f"[{protocol}] train∩test leakage: {list(tt)[:5]}"
    assert len(vt) == 0, f"[{protocol}] val∩test leakage: {list(vt)[:5]}"


def test_split_all_partitions_non_empty():
    """All three partitions should have at least one commit."""
    for protocol in ["P1", "P2", "P3"]:
        split = _load_split_if_exists(protocol)
        if split is None:
            pytest.skip(f"Split {protocol} not found")
        for partition in ["train", "val", "test"]:
            assert len(split[partition]) > 0, f"[{protocol}] {partition} is empty"


# ---------------------------------------------------------------------------
# Smoke test: predict
# ---------------------------------------------------------------------------

def test_predict_smoke():
    """predict_one should run without error (uses CPU if no GPU, skips if no model)."""
    ckpt = ROOT / "models" / "best_B4_P1.pt"
    if not ckpt.exists():
        pytest.skip("Best model checkpoint not found — run training first")

    import torch
    import yaml
    from vulnreview.predict import load_model_and_tokenizer, predict_one

    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    model, tokenizer, label_maps = load_model_and_tokenizer(cfg, ROOT / "models", device)

    text = "void foo() {\n    char buf[8];\n    gets(buf);\n}"
    result = predict_one(text, model, tokenizer, label_maps, cfg, device)

    assert "vuln_probability" in result
    assert 0.0 <= result["vuln_probability"] <= 1.0
    assert result["vuln_prediction"] in ("VULNERABLE", "FIXED")
    assert "cwe_id" in result
    assert "severity" in result

