"""
model.py - Multi-task CodeBERT model for vulnerability detection.

Architecture:
  - Shared encoder: microsoft/codebert-base
  - First-token ([CLS]) pooling -> dropout(0.1)
  - Three linear heads:
      1. vuln head: 2 classes (fixed=0, vulnerable=1)
      2. CWE head: 11 classes (top-10 + Other)
      3. severity head: 3 classes (low, medium, high)
  - Loss = L_vuln + 1[y_vuln=1] * (0.5 * L_cwe + 0.5 * L_sev)
  - class-weighted CE, masked labels (=-100) ignored
"""
from __future__ import annotations
import logging
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

MASKED_LABEL = -100


def compute_class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    from collections import Counter
    counts = Counter(l for l in labels if l != MASKED_LABEL)
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights.append(total / (num_classes * cnt))
    return torch.tensor(weights, dtype=torch.float, device=device)


class MultiTaskCodeBERT(nn.Module):
    """
    Multi-task CodeBERT: shared encoder + 3 heads (vuln, CWE, severity).
    Supports ablation: disable CWE head and/or severity head.
    """
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_cwe_classes: int = 11,
        num_severity_classes: int = 3,
        dropout: float = 0.1,
        use_cwe_head: bool = True,
        use_sev_head: bool = True,
        vuln_class_weights: Optional[torch.Tensor] = None,
        cwe_class_weights: Optional[torch.Tensor] = None,
        sev_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.use_cwe_head = use_cwe_head
        self.use_sev_head = use_sev_head

        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size  # 768 for codebert-base

        self.dropout = nn.Dropout(dropout)
        self.vuln_head = nn.Linear(hidden_size, 2)
        if use_cwe_head:
            self.cwe_head = nn.Linear(hidden_size, num_cwe_classes)
        if use_sev_head:
            self.sev_head = nn.Linear(hidden_size, num_severity_classes)

        # Register class weights as buffers
        if vuln_class_weights is not None:
            self.register_buffer("vuln_weights", vuln_class_weights)
        else:
            self.register_buffer("vuln_weights", torch.ones(2))

        if use_cwe_head:
            if cwe_class_weights is not None:
                self.register_buffer("cwe_weights", cwe_class_weights)
            else:
                self.register_buffer("cwe_weights", torch.ones(num_cwe_classes))

        if use_sev_head:
            if sev_class_weights is not None:
                self.register_buffer("sev_weights", sev_class_weights)
            else:
                self.register_buffer("sev_weights", torch.ones(num_severity_classes))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        vuln_labels: Optional[torch.Tensor] = None,
        cwe_labels: Optional[torch.Tensor] = None,
        sev_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        cls_output = self.dropout(cls_output)

        # Logits
        vuln_logits = self.vuln_head(cls_output)
        cwe_logits = self.cwe_head(cls_output) if self.use_cwe_head else None
        sev_logits = self.sev_head(cls_output) if self.use_sev_head else None

        result = {
            "vuln_logits": vuln_logits,
            "cwe_logits": cwe_logits,
            "sev_logits": sev_logits,
            "last_hidden_state": outputs.last_hidden_state,
        }

        if vuln_labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.vuln_weights, ignore_index=MASKED_LABEL)
            l_vuln = loss_fct(vuln_logits, vuln_labels)

            l_aux = torch.tensor(0.0, device=vuln_logits.device)
            if self.use_cwe_head and cwe_labels is not None:
                # Only on before samples (vuln=1 in labels)
                cwe_fct = CrossEntropyLoss(weight=self.cwe_weights, ignore_index=MASKED_LABEL)
                l_cwe = cwe_fct(cwe_logits, cwe_labels)
                l_aux = l_aux + 0.5 * l_cwe

            if self.use_sev_head and sev_labels is not None:
                sev_fct = CrossEntropyLoss(weight=self.sev_weights, ignore_index=MASKED_LABEL)
                l_sev = sev_fct(sev_logits, sev_labels)
                l_aux = l_aux + 0.5 * l_sev

            # Only add aux loss for vulnerable samples
            if vuln_labels is not None:
                vuln_mask = (vuln_labels == 1).float()
                # Scale aux loss by proportion of vulnerable samples
                vuln_ratio = vuln_mask.mean()
                total_loss = l_vuln + vuln_ratio * l_aux
            else:
                total_loss = l_vuln

            result["loss"] = total_loss
            result["vuln_loss"] = l_vuln
            result["aux_loss"] = l_aux

        return result


class SingleTaskCodeBERT(nn.Module):
    """Single-task CodeBERT: vuln head only (B3 baseline)."""
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        dropout: float = 0.1,
        vuln_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.vuln_head = nn.Linear(hidden_size, 2)

        if vuln_class_weights is not None:
            self.register_buffer("vuln_weights", vuln_class_weights)
        else:
            self.register_buffer("vuln_weights", torch.ones(2))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        vuln_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        vuln_logits = self.vuln_head(cls_output)

        result = {
            "vuln_logits": vuln_logits,
            "cwe_logits": None,
            "sev_logits": None,
            "last_hidden_state": outputs.last_hidden_state,
        }

        if vuln_labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.vuln_weights, ignore_index=MASKED_LABEL)
            result["loss"] = loss_fct(vuln_logits, vuln_labels)
        return result
