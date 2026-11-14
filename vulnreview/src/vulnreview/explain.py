"""
explain.py - Integrated Gradients explanation for VulnReview.

Uses captum to compute token attributions, then maps them to source lines.
Saves qualitative examples and computes Loc@1 / Loc@3.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MASKED_LABEL = -100


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_token_attributions(
    model,
    tokenizer,
    text: str,
    target_class: int = 1,
    n_steps: int = 30,
    device: torch.device = torch.device("cpu"),
    max_len: int = 512,
) -> tuple[list[str], list[float]]:
    """
    Compute Integrated Gradients for the input text w.r.t. the embedding layer.
    Returns (tokens, attributions) aligned.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        logger.error("captum not installed. Install with: pip install captum")
        return [], []

    model.eval()
    enc = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    # Get embedding layer
    embedding_layer = model.encoder.embeddings.word_embeddings

    def forward_func(embeddings):
        # Attention mask shape must match batch size of interpolated embeddings
        bsz = embeddings.shape[0]
        att_mask = attention_mask.expand(bsz, -1)
        tt_ids = token_type_ids.expand(bsz, -1) if token_type_ids is not None else None
        outputs = model.encoder(
            inputs_embeds=embeddings,
            attention_mask=att_mask,
            token_type_ids=tt_ids,
        )
        cls_out = model.dropout(outputs.last_hidden_state[:, 0, :])
        logits = model.vuln_head(cls_out)
        return torch.softmax(logits, dim=-1)[:, target_class]

    embeddings = embedding_layer(input_ids)
    baseline = torch.zeros_like(embeddings)

    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(
        embeddings,
        baselines=baseline,
        target=None,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    attr_scores = attributions.squeeze(0).sum(dim=-1).abs().detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    mask = attention_mask.squeeze(0).cpu().numpy()
    real_len = int(mask.sum())
    tokens = tokens[:real_len]
    attr_scores = attr_scores[:real_len]

    return tokens, attr_scores.tolist()


def tokens_to_line_scores(
    tokens: list[str],
    attr_scores: list[float],
    source_text: str,
) -> dict[int, float]:
    """
    Map token attributions back to source lines.
    Returns {line_idx: score} where score = sum of positive token attributions on that line.
    """
    if not tokens or not attr_scores:
        return {}

    source_lines = source_text.splitlines()
    n_lines = len(source_lines)
    if n_lines == 0:
        return {}

    line_scores: dict[int, float] = {i: 0.0 for i in range(n_lines)}

    char_pos = 0
    tok_idx = 0
    for line_idx, line in enumerate(source_lines):
        line_len = len(line) + 1
        line_end = char_pos + line_len
        while tok_idx < len(tokens):
            tok = tokens[tok_idx]
            tok_clean = tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
            tok_end = char_pos + len(tok_clean)
            if tok_end > line_end:
                break
            score = max(0.0, attr_scores[tok_idx])
            line_scores[line_idx] = line_scores.get(line_idx, 0.0) + score
            char_pos = tok_end
            tok_idx += 1
        char_pos = line_end

    return line_scores


def loc_at_k(
    line_scores: dict[int, float],
    removed_indices: list[int],
    k: int = 1,
) -> bool:
    """Return True if at least one of top-k scored lines is a removed line."""
    if not line_scores or not removed_indices:
        return False
    sorted_lines = sorted(line_scores.items(), key=lambda x: -x[1])
    top_k = {l for l, _ in sorted_lines[:k]}
    return bool(top_k & set(removed_indices))


def run_explanation(
    protocol: str,
    max_samples: int,
    cfg: dict,
    root: Path,
    quick: bool = False,
) -> dict:
    """Run IG explanation on test set, compute Loc@1 / Loc@3."""
    import sys
    sys.path.insert(0, str(root / "src"))

    from vulnreview.model import MultiTaskCodeBERT
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "_quick" if quick else ""
    processed_dir = root / cfg["data"]["processed_dir"]
    splits_dir = root / cfg["results"]["splits_dir"]
    models_dir = root / cfg["results"]["models_dir"]
    results_dir = root / cfg["results"]["dir"]

    ckpt_path = models_dir / "best_B4_P1.pt"
    if not ckpt_path.exists():
        logger.error("Best model checkpoint not found at %s. Run training first.", ckpt_path)
        return {}

    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultiTaskCodeBERT(
        model_name=cfg["model"]["name"],
        num_cwe_classes=cfg["model"]["num_cwe_classes"],
        num_severity_classes=cfg["model"]["num_severity_classes"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    samples_df = pd.read_csv(processed_dir / f"samples{suffix}.csv")
    split_path = splits_dir / f"{protocol}{suffix}.json"
    with open(split_path) as f:
        split = json.load(f)
    test_ids = set(split["test"])
    test_df = samples_df[
        (samples_df["commit_id"].isin(test_ids)) &
        (samples_df["sample_type"] == "before") &
        (samples_df["vuln"] == 1)
    ].copy()

    if len(test_df) > max_samples:
        test_df = test_df.sample(n=max_samples, random_state=42)
    logger.info("Running IG on %d vulnerable test samples...", len(test_df))

    loc1_hits = 0
    loc3_hits = 0
    n_valid = 0
    examples = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Explaining"):
        text = str(row["text"])
        try:
            rli = json.loads(str(row.get("removed_line_indices", "[]")))
        except Exception:
            rli = []

        if not rli:
            continue

        tokens, attr_scores = get_token_attributions(
            model, tokenizer, text,
            target_class=1,
            n_steps=cfg["explain"]["ig_steps"],
            device=device,
            max_len=cfg["model"].get("max_len", 512),
        )
        if not tokens:
            continue

        line_scores = tokens_to_line_scores(tokens, attr_scores, text)
        l1 = loc_at_k(line_scores, rli, k=1)
        l3 = loc_at_k(line_scores, rli, k=3)
        loc1_hits += int(l1)
        loc3_hits += int(l3)
        n_valid += 1

        if len(examples) < 5:
            sorted_lines = sorted(line_scores.items(), key=lambda x: -x[1])
            source_lines = text.splitlines()
            examples.append({
                "cve_id": row.get("cve_id", ""),
                "cwe_id": row.get("cwe_id", ""),
                "text": text,
                "removed_indices": rli,
                "top5_lines": [(i, source_lines[i] if i < len(source_lines) else "?", s)
                               for i, s in sorted_lines[:5]],
                "loc1": l1,
                "loc3": l3,
            })

    loc1 = loc1_hits / n_valid if n_valid > 0 else 0.0
    loc3 = loc3_hits / n_valid if n_valid > 0 else 0.0
    logger.info("Protocol %s: Loc@1=%.4f Loc@3=%.4f (n=%d)", protocol, loc1, loc3, n_valid)

    expl_path = results_dir / "explanations.md"
    with open(expl_path, "w", encoding="utf-8") as f:
        f.write("# Explanation Examples (Integrated Gradients)\n\n")
        for i, ex in enumerate(examples, 1):
            f.write(f"## Example {i}: {ex['cve_id']} ({ex['cwe_id']})\n\n")
            f.write("**Top suspicious lines:**\n\n")
            f.write("```c\n")
            for line_idx, line_text, score in ex["top5_lines"]:
                marker = ">>>" if line_idx in ex["removed_indices"] else "   "
                f.write(f"{marker} [{line_idx:3d}] (score={score:.4f}) {line_text}\n")
            f.write("```\n\n")
            f.write(f"- Loc@1: {'✓' if ex['loc1'] else '✗'}  Loc@3: {'✓' if ex['loc3'] else '✗'}\n\n")
            f.write("---\n\n")

    expl_metrics = {
        "protocol": protocol,
        "loc1": loc1,
        "loc3": loc3,
        "n_explained": n_valid,
    }
    expl_metrics_path = results_dir / f"explain_metrics_{protocol}.json"
    with open(expl_metrics_path, "w") as f:
        json.dump(expl_metrics, f, indent=2)

    return expl_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--protocol", default="P1")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(".")
    metrics = run_explanation(args.protocol, args.max_samples, cfg, root, args.quick)
    if metrics:
        print(f"\nLoc@1: {metrics['loc1']:.4f}  Loc@3: {metrics['loc3']:.4f}  (n={metrics['n_explained']})")


if __name__ == "__main__":
    main()

