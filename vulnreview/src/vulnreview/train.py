"""
train.py - Training loop for VulnReview models.

Supports:
- B3: single-task CodeBERT
- B4: multi-task CodeBERT (full, ablation no_cwe, ablation no_sev)
- Protocols P1, P2, P3
- Multiple seeds
- Pair-aware batching
- AdamW + linear warmup/decay
- fp16 on GPU
- Early stopping (patience=3 on val F1)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

from vulnreview.model import MultiTaskCodeBERT, SingleTaskCodeBERT, compute_class_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MASKED_LABEL = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VulnDataset(Dataset):
    """Dataset wrapping tokenized pairs."""
    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        max_len: int = 512,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "vuln_labels": torch.tensor(s["vuln"], dtype=torch.long),
            "cwe_labels": torch.tensor(s.get("cwe_label", MASKED_LABEL), dtype=torch.long),
            "sev_labels": torch.tensor(s.get("sev_label", MASKED_LABEL), dtype=torch.long),
            "pair_id": s["pair_id"],
            "sample_type": s["sample_type"],
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


class PairAwareSampler(Sampler):
    """
    Pair-aware batching: ensure both samples of a pair appear in the same batch.
    Groups pairs by pair_id, then yields batches of size batch_size.
    """
    def __init__(self, dataset: VulnDataset, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        # Group indices by pair_id
        pair_to_indices: dict[int, list[int]] = {}
        for i, s in enumerate(dataset.samples):
            pid = s["pair_id"]
            pair_to_indices.setdefault(pid, []).append(i)
        self.pairs = list(pair_to_indices.values())

    def __iter__(self):
        rng = random.Random(self.seed)
        pairs = self.pairs[:]
        rng.shuffle(pairs)
        # Flatten into indices, keeping pairs together
        all_indices = []
        for pair in pairs:
            all_indices.extend(pair)
        # Yield in chunks of batch_size
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i + self.batch_size]

    def __len__(self):
        return (sum(len(p) for p in self.pairs) + self.batch_size - 1) // self.batch_size


def collate_fn(batch: list[dict]) -> dict:
    keys = ["input_ids", "attention_mask", "vuln_labels", "cwe_labels", "sev_labels"]
    if "token_type_ids" in batch[0]:
        keys.append("token_type_ids")
    result = {k: torch.stack([b[k] for b in batch]) for k in keys}
    result["pair_ids"] = [b["pair_id"] for b in batch]
    result["sample_types"] = [b["sample_type"] for b in batch]
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_proba, test_df: pd.DataFrame) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Pairwise accuracy
    df = test_df.copy()
    df["pred_proba"] = y_proba
    pair_acc = 0.0
    n_pairs = 0
    for pid, group in df.groupby("pair_id"):
        before = group[group["sample_type"] == "before"]
        after = group[group["sample_type"] == "after"]
        if len(before) == 1 and len(after) == 1:
            if before["pred_proba"].values[0] > after["pred_proba"].values[0]:
                pair_acc += 1
            n_pairs += 1
    pair_acc = pair_acc / n_pairs if n_pairs > 0 else 0.0

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "pair_accuracy": pair_acc}


def compute_cwe_sev_metrics(
    test_df: pd.DataFrame,
    cwe_preds: list[int],
    sev_preds: list[int],
) -> dict:
    """Compute CWE macro-F1, severity macro-F1 and mean absolute band error."""
    # Only on before samples with non-masked labels
    before_mask = (test_df["sample_type"] == "before").values
    cwe_true = test_df["cwe_label"].values[before_mask]
    sev_true = test_df["sev_label"].values[before_mask]
    cwe_pred = np.array(cwe_preds)[before_mask]
    sev_pred = np.array(sev_preds)[before_mask]

    # Mask out -100
    cwe_valid = cwe_true != MASKED_LABEL
    sev_valid = sev_true != MASKED_LABEL

    cwe_f1 = f1_score(cwe_true[cwe_valid], cwe_pred[cwe_valid], average="macro", zero_division=0) if cwe_valid.any() else 0.0
    sev_f1 = f1_score(sev_true[sev_valid], sev_pred[sev_valid], average="macro", zero_division=0) if sev_valid.any() else 0.0
    sev_mae = float(np.mean(np.abs(sev_true[sev_valid] - sev_pred[sev_valid]))) if sev_valid.any() else 0.0

    return {"cwe_macro_f1": cwe_f1, "sev_macro_f1": sev_f1, "sev_mae": sev_mae}


def compute_severity_weighted_recall(test_df: pd.DataFrame, y_pred: np.ndarray) -> float:
    """Severity-weighted recall = sum(score_i * 1[pred=vuln]) / sum(score_i) on before samples with score."""
    before_mask = (test_df["sample_type"] == "before").values
    scores = pd.to_numeric(test_df["score"], errors="coerce").values
    has_score = ~np.isnan(scores) & before_mask
    if not has_score.any():
        return 0.0
    s = scores[has_score]
    p = (y_pred[has_score] == 1).astype(float)
    return float(np.sum(s * p) / np.sum(s))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    grad_accum: int = 1,
    use_fp16: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        vuln_labels = batch["vuln_labels"].to(device)
        cwe_labels = batch["cwe_labels"].to(device)
        sev_labels = batch["sev_labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with autocast(enabled=use_fp16):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                vuln_labels=vuln_labels,
                cwe_labels=cwe_labels,
                sev_labels=sev_labels,
            )
            loss = out["loss"] / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += out["loss"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    df: pd.DataFrame,
    use_fp16: bool = True,
) -> dict:
    model.eval()
    all_vuln_logits = []
    all_cwe_preds = []
    all_sev_preds = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with autocast(enabled=use_fp16):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        all_vuln_logits.append(out["vuln_logits"].float().cpu())
        if out["cwe_logits"] is not None:
            all_cwe_preds.extend(out["cwe_logits"].float().argmax(dim=-1).cpu().tolist())
        else:
            all_cwe_preds.extend([0] * len(batch["vuln_labels"]))
        if out["sev_logits"] is not None:
            all_sev_preds.extend(out["sev_logits"].float().argmax(dim=-1).cpu().tolist())
        else:
            all_sev_preds.extend([0] * len(batch["vuln_labels"]))

    all_vuln_logits = torch.cat(all_vuln_logits, dim=0)
    proba = torch.softmax(all_vuln_logits, dim=-1)[:, 1].numpy()
    preds = (proba >= 0.5).astype(int)
    true = df["vuln"].values

    main_metrics = compute_metrics(true, preds, proba, df)
    aux_metrics = compute_cwe_sev_metrics(df, all_cwe_preds, all_sev_preds)
    sw_rec = compute_severity_weighted_recall(df, preds)

    return {**main_metrics, **aux_metrics, "sw_recall": sw_rec}


def load_split_samples(
    samples_df: pd.DataFrame,
    split: dict,
    split_name: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])

    def to_records(df_subset: pd.DataFrame) -> list[dict]:
        records = []
        for _, row in df_subset.iterrows():
            d = row.to_dict()
            # Parse removed_line_indices back from JSON string
            rli = d.get("removed_line_indices", "[]")
            if isinstance(rli, str):
                try:
                    d["removed_line_indices"] = json.loads(rli)
                except Exception:
                    d["removed_line_indices"] = []
            records.append(d)
        return records

    train_df = samples_df[samples_df["commit_id"].isin(train_ids)]
    val_df = samples_df[samples_df["commit_id"].isin(val_ids)]
    test_df = samples_df[samples_df["commit_id"].isin(test_ids)]

    return to_records(train_df), to_records(val_df), to_records(test_df)


def run_training(
    model_type: str,       # "B3" or "B4"
    ablation: str,         # "none", "no_cwe", "no_sev"
    protocol: str,         # "P1", "P2", "P3"
    seed: int,
    cfg: dict,
    root: Path,
    quick: bool = False,
) -> dict:
    """Run one training configuration. Returns results dict."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    use_fp16 = device.type == "cuda" and cfg["training"]["fp16"]

    suffix = "_quick" if quick else ""
    processed_dir = root / cfg["data"]["processed_dir"]
    splits_dir = root / cfg["results"]["splits_dir"]
    models_dir = root / cfg["results"]["models_dir"]
    results_dir = root / cfg["results"]["dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    samples_path = processed_dir / f"samples{suffix}.csv"
    split_path = splits_dir / f"{protocol}{suffix}.json"

    samples_df = pd.read_csv(samples_path)
    with open(split_path) as f:
        split = json.load(f)

    train_records, val_records, test_records = load_split_samples(samples_df, split, protocol)
    logger.info("Protocol %s: train=%d, val=%d, test=%d samples",
                protocol, len(train_records), len(val_records), len(test_records))

    model_name = cfg["model"]["name"]
    max_len = cfg["model"].get("max_len", 512)
    if quick:
        max_len = 256

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Build datasets
    train_ds = VulnDataset(train_records, tokenizer, max_len)
    val_ds = VulnDataset(val_records, tokenizer, max_len)
    test_ds = VulnDataset(test_records, tokenizer, max_len)

    batch_size = cfg["training"]["batch_size"]
    grad_accum = cfg["training"]["grad_accum_steps"]

    # Pair-aware batching for train; standard for val/test
    train_sampler = PairAwareSampler(train_ds, batch_size, seed)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Compute class weights from training labels
    train_vuln_labels = [r["vuln"] for r in train_records]
    vuln_weights = compute_class_weights(train_vuln_labels, 2, device)

    # Build model
    if model_type == "B3":
        model = SingleTaskCodeBERT(
            model_name=model_name,
            dropout=cfg["model"]["dropout"],
            vuln_class_weights=vuln_weights,
        )
        model_label = "B3_single_task"
    else:  # B4
        use_cwe = (ablation != "no_cwe")
        use_sev = (ablation != "no_sev")
        cwe_labels_list = [r.get("cwe_label", MASKED_LABEL) for r in train_records]
        sev_labels_list = [r.get("sev_label", MASKED_LABEL) for r in train_records]
        cwe_weights = compute_class_weights(cwe_labels_list, cfg["model"]["num_cwe_classes"], device) if use_cwe else None
        sev_weights = compute_class_weights(sev_labels_list, cfg["model"]["num_severity_classes"], device) if use_sev else None

        model = MultiTaskCodeBERT(
            model_name=model_name,
            num_cwe_classes=cfg["model"]["num_cwe_classes"],
            num_severity_classes=cfg["model"]["num_severity_classes"],
            dropout=cfg["model"]["dropout"],
            use_cwe_head=use_cwe,
            use_sev_head=use_sev,
            vuln_class_weights=vuln_weights,
            cwe_class_weights=cwe_weights,
            sev_class_weights=sev_weights,
        )
        model_label = f"B4_{ablation}" if ablation != "none" else "B4_full"

    model.to(device)

    # Optimizer
    epochs = 1 if quick else cfg["training"]["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=use_fp16)

    # Training loop
    best_val_f1 = -1.0
    best_epoch = 0
    patience = cfg["training"]["early_stopping_patience"]
    patience_counter = 0
    best_state = None
    t_start = time.time()

    val_df = pd.DataFrame(val_records)
    test_df = pd.DataFrame(test_records)

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d/%d [%s %s %s seed=%d]", epoch, epochs, model_label, protocol, suffix, seed)
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, grad_accum, use_fp16)

        val_metrics = evaluate(model, val_loader, device, val_df, use_fp16)
        val_f1 = val_metrics["f1"]
        logger.info("  train_loss=%.4f val_f1=%.4f val_pair_acc=%.4f", train_loss, val_f1, val_metrics["pair_accuracy"])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    train_time = time.time() - t_start

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Test evaluation
    test_metrics = evaluate(model, test_loader, device, test_df, use_fp16)
    logger.info("TEST [%s %s seed=%d]: %s", model_label, protocol, seed, test_metrics)

    # Save checkpoint if this is the best B4 P1 seed=0
    is_best = (model_type == "B4" and ablation == "none" and protocol == "P1" and seed == 0)
    if is_best:
        ckpt_path = models_dir / "best_B4_P1.pt"
        torch.save({
            "model_state_dict": best_state if best_state else model.state_dict(),
            "model_type": model_label,
            "protocol": protocol,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "config": cfg,
        }, ckpt_path)
        logger.info("Saved best checkpoint to %s", ckpt_path)

    result = {
        "model": model_label,
        "model_type": model_type,
        "ablation": ablation,
        "protocol": protocol,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "train_time_sec": train_time,
        "quick": quick,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VulnReview models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--protocol", default="P1", choices=["P1", "P2", "P3"])
    parser.add_argument("--model", default="B4", choices=["B3", "B4"])
    parser.add_argument("--ablation", default="none", choices=["none", "no_cwe", "no_sev"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs in config")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    root = Path(".")
    all_results = []

    for seed in args.seeds:
        result = run_training(
            model_type=args.model,
            ablation=args.ablation,
            protocol=args.protocol,
            seed=seed,
            cfg=cfg,
            root=root,
            quick=args.quick,
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    suffix = "_quick" if args.quick else ""
    out_path = root / cfg["results"]["dir"] / f"train_results_{args.model}_{args.ablation}_{args.protocol}{suffix}.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Saved training results to %s", out_path)
    print("\nResults:")
    print(results_df[["model", "protocol", "seed", "test_f1", "test_pair_accuracy", "train_time_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
