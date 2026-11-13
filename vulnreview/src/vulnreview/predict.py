"""
predict.py - CLI inference for VulnReview.

Usage:
    python -m vulnreview.predict --file example.c
    python -m vulnreview.predict --text "int foo() { char buf[8]; gets(buf); }"
    python -m vulnreview.predict --diff hunk.diff
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CWE_NAMES = {
    "CWE-119": "Improper Restriction of Operations within Memory Buffer",
    "CWE-125": "Out-of-bounds Read",
    "CWE-787": "Out-of-bounds Write",
    "CWE-416": "Use After Free",
    "CWE-476": "NULL Pointer Dereference",
    "CWE-190": "Integer Overflow or Wraparound",
    "CWE-20":  "Improper Input Validation",
    "CWE-200": "Information Exposure",
    "CWE-399": "Resource Management Errors",
    "CWE-264": "Permissions, Privileges, and Access Controls",
    "Other":   "Other / Unknown",
}

SEVERITY_NAMES = ["Low", "Medium", "High"]


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(cfg: dict, models_dir: Path, device: torch.device):
    """Load best B4 checkpoint and tokenizer. Returns (model, tokenizer, label_maps)."""
    from transformers import AutoTokenizer
    from vulnreview.model import MultiTaskCodeBERT

    ckpt_path = models_dir / "best_B4_P1.pt"
    labels_path = models_dir / "labels.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {ckpt_path}. "
            "Run training first: python -m vulnreview.train --protocol P1 --model B4 --seeds 0"
        )

    logger.info("Loading checkpoint from %s ...", ckpt_path)
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

    label_maps = {}
    if labels_path.exists():
        with open(labels_path) as f:
            label_maps = json.load(f)

    return model, tokenizer, label_maps


@torch.no_grad()
def predict_one(
    text: str,
    model,
    tokenizer,
    label_maps: dict,
    cfg: dict,
    device: torch.device,
    n_ig_steps: int = 30,
) -> dict:
    """
    Predict vulnerability, CWE, and severity for one code snippet.
    Returns dict with probabilities, predictions, and line scores.
    """
    max_len = cfg["model"].get("max_len", 512)
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

    model.eval()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    vuln_proba = torch.softmax(out["vuln_logits"], dim=-1)[0].cpu().tolist()
    vuln_pred = int(torch.argmax(out["vuln_logits"], dim=-1).item())

    cwe_pred_idx = None
    cwe_proba = None
    cwe_name = "N/A"
    cwe_id = "N/A"
    if out["cwe_logits"] is not None:
        cwe_proba_t = torch.softmax(out["cwe_logits"], dim=-1)[0].cpu().tolist()
        cwe_pred_idx = int(torch.argmax(out["cwe_logits"], dim=-1).item())
        cwe_proba = cwe_proba_t
        ordered_cwes = label_maps.get("ordered_cwes", [])
        if ordered_cwes and cwe_pred_idx < len(ordered_cwes):
            cwe_id = ordered_cwes[cwe_pred_idx]
            cwe_name = CWE_NAMES.get(cwe_id, cwe_id)

    sev_pred_idx = None
    sev_proba = None
    sev_name = "N/A"
    if out["sev_logits"] is not None:
        sev_proba_t = torch.softmax(out["sev_logits"], dim=-1)[0].cpu().tolist()
        sev_pred_idx = int(torch.argmax(out["sev_logits"], dim=-1).item())
        sev_proba = sev_proba_t
        sev_name = SEVERITY_NAMES[sev_pred_idx]

    source_lines = text.splitlines()
    line_scores = {i: 0.0 for i in range(len(source_lines))}

    return {
        "vuln_probability": vuln_proba[1],
        "fixed_probability": vuln_proba[0],
        "vuln_prediction": "VULNERABLE" if vuln_pred == 1 else "FIXED",
        "cwe_id": cwe_id,
        "cwe_name": cwe_name,
        "cwe_confidence": float(max(cwe_proba)) if cwe_proba else None,
        "severity": sev_name,
        "severity_probabilities": dict(zip(SEVERITY_NAMES, sev_proba)) if sev_proba else {},
        "line_scores": line_scores,
        "source_lines": source_lines,
    }


def print_prediction(result: dict) -> None:
    """Pretty-print prediction result."""
    print("\n" + "=" * 60)
    print("VulnReview Prediction")
    print("=" * 60)
    print(f"  Verdict     : {result['vuln_prediction']}")
    print(f"  Vuln prob   : {result['vuln_probability']:.4f}")
    print(f"  Fixed prob  : {result['fixed_probability']:.4f}")
    print(f"  CWE type    : {result['cwe_id']} - {result['cwe_name']}")
    print(f"  CWE conf    : {result['cwe_confidence']:.4f}" if result["cwe_confidence"] else "  CWE conf    : N/A")
    print(f"  Severity    : {result['severity']}")
    if result["severity_probabilities"]:
        for s, p in result["severity_probabilities"].items():
            print(f"    {s:8s}: {p:.4f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VulnReview: predict vulnerability in C/C++ code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m vulnreview.predict --file example.c
  python -m vulnreview.predict --text "gets(buf);"
  python -m vulnreview.predict --diff hunk.patch
""",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--file", help="Path to C/C++ source file")
    parser.add_argument("--text", help="Inline code snippet")
    parser.add_argument("--diff", help="Path to unified diff file")
    parser.add_argument("--output-json", help="Save JSON output to this path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    root = Path(".")
    models_dir = root / cfg["results"]["models_dir"]

    try:
        model, tokenizer, label_maps = load_model_and_tokenizer(cfg, models_dir, device)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    elif args.text:
        text = args.text
    elif args.diff:
        text = Path(args.diff).read_text(encoding="utf-8", errors="replace")
    else:
        parser.print_help()
        return

    result = predict_one(text, model, tokenizer, label_maps, cfg, device)
    print_prediction(result)

    if args.output_json:
        out = {k: v for k, v in result.items() if k not in ("source_lines", "line_scores")}
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        logger.info("Saved JSON output to %s", args.output_json)


if __name__ == "__main__":
    main()

