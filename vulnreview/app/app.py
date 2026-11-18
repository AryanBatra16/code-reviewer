"""
app.py - VulnReview Streamlit Demo Application.

Run with: streamlit run app/app.py

Features:
- Text box / file upload for C/C++ snippet or unified diff
- Shows vuln probability bar, CWE type + confidence, severity band
- Code with top-k suspicious lines highlighted
- 3 built-in example snippets
- Sidebar: top-k selector, model info (from results files)
- Graceful degradation if model missing or no GPU
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import torch
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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

SEVERITY_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

# ---------------------------------------------------------------------------
# Config and model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading VulnReview model...")
def load_model_cached(config_path: str):
    """Load model and tokenizer, cached across Streamlit re-runs."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        from vulnreview.model import MultiTaskCodeBERT
        from transformers import AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models_dir = ROOT / cfg["results"]["models_dir"]
        ckpt_path = models_dir / "best_B4_P1.pt"
        labels_path = models_dir / "labels.json"

        if not ckpt_path.exists():
            return None, None, None, cfg, str(ckpt_path), device

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

        return model, tokenizer, label_maps, cfg, str(ckpt_path), device

    except Exception as e:
        logger.error("Model load failed: %s", e, exc_info=True)
        return None, None, None, {}, "", torch.device("cpu")


def get_test_f1_from_results(results_dir: Path) -> str:
    """Read test F1 from results CSV, never hard-code."""
    try:
        import pandas as pd
        for f in results_dir.glob("train_results_B4_none_P1*.csv"):
            df = pd.read_csv(f)
            if "test_f1" in df.columns:
                mean_f1 = df["test_f1"].mean()
                return f"{mean_f1:.4f}"
    except Exception:
        pass
    return "not yet computed"


@torch.no_grad()
def predict(text: str, model, tokenizer, label_maps: dict, cfg: dict, device) -> dict:
    """Run inference on text. Returns prediction dict."""
    from vulnreview.predict import predict_one
    return predict_one(text, model, tokenizer, label_maps, cfg, device)


def highlight_code_html(source_lines: list[str], line_scores: dict[int, float], top_k: int) -> str:
    """Generate HTML with top-k lines highlighted by attribution score intensity."""
    if not line_scores or not source_lines:
        escaped = "\n".join(
            f'<span style="font-family:monospace">{_escape(l)}</span>'
            for l in source_lines
        )
        return f"<pre>{escaped}</pre>"

    max_score = max(line_scores.values()) if line_scores else 1.0
    sorted_lines = sorted(line_scores.items(), key=lambda x: -x[1])
    top_k_set = {l for l, _ in sorted_lines[:top_k]}

    html_lines = []
    for i, line in enumerate(source_lines):
        score = line_scores.get(i, 0.0)
        if i in top_k_set and max_score > 0:
            intensity = int(255 * (1 - score / max_score) * 0.7 + 255 * 0.3)
            bg = f"rgb(255, {intensity}, {intensity})"
            html_lines.append(
                f'<div style="background:{bg};font-family:monospace;white-space:pre">'
                f'<b>&#9888; {i+1:4d} &boxv;</b> {_escape(line)}</div>'
            )
        else:
            html_lines.append(
                f'<div style="font-family:monospace;white-space:pre">'
                f'  {i+1:4d} &boxv; {_escape(line)}</div>'
            )
    return "".join(html_lines)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="VulnReview",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 VulnReview")
    st.subheader("Patch-Pair Multi-Task Learning for Vulnerability-Aware Code Review")

    # Disclaimer
    st.warning(
        "⚠️ **Research Prototype**: Trained on Big-Vul dataset (94.8% C code, dominated by Chrome/Linux/Android projects). "
        "This is NOT a substitute for a professional security audit. Results may not generalize to other codebases or languages."
    )

    # Load model
    config_path = str(ROOT / "configs" / "config.yaml")
    model, tokenizer, label_maps, cfg, ckpt_path, device = load_model_cached(config_path)
    results_dir = ROOT / cfg.get("results", {}).get("dir", "results")
    test_f1 = get_test_f1_from_results(results_dir)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Top-K suspicious lines to highlight", 1, 10, 3)
        st.divider()
        st.header("ℹ️ Model Info")
        st.markdown(f"**Checkpoint**: `{Path(ckpt_path).name if ckpt_path else 'Not loaded'}`")
        st.markdown(f"**Base model**: `microsoft/codebert-base`")
        st.markdown(f"**Training protocol**: P1 (Random, commit-grouped)")
        st.markdown(f"**Test F1**: {test_f1}")
        st.markdown(f"**Device**: `{device}`")
        st.divider()
        st.header("📚 About")
        st.markdown(
            "Built on [Big-Vul](https://github.com/ZeoVan/MSR_20_Code_vulnerability_csv_Dataset) "
            "(Fan et al., MSR 2020) and [CodeBERT](https://arxiv.org/abs/2002.08155) "
            "(Feng et al., 2020).\n\n"
            "**Author**: Aryan Batra (24CSU027), The NorthCap University"
        )

    # Model status
    if model is None:
        st.error(
            f"❌ Model checkpoint not found at: `{ckpt_path}`\n\n"
            "**To train the model, run:**\n"
            "```bash\n"
            "cd vulnreview\n"
            "python -m vulnreview.data_prep\n"
            "python -m vulnreview.train --protocol P1 --model B4 --seeds 0 1 2\n"
            "```"
        )
        st.info("You can still browse the built-in examples below.")

    # Input section
    st.header("📝 Input")
    col1, col2 = st.columns([3, 1])
    with col1:
        input_mode = st.radio("Input mode", ["Type / paste code", "Upload file"], horizontal=True)

    code_input = ""

    if input_mode == "Upload file":
        uploaded = st.file_uploader("Upload a C/C++ file or .diff file", type=["c", "cpp", "cc", "h", "diff", "patch"])
        if uploaded:
            code_input = uploaded.read().decode("utf-8", errors="replace")
            st.code(code_input, language="c")

    else:
        code_input = st.text_area(
            "Paste C/C++ snippet or unified-diff hunk here:",
            height=250,
            placeholder="void foo() {\n    char buf[8];\n    gets(buf);  // vulnerable!\n}",
        )

    # Analyze button
    if st.button("🔍 Analyze", type="primary", disabled=(model is None or not code_input.strip())):
        with st.spinner("Running model inference..."):
            try:
                result = predict(code_input, model, tokenizer, label_maps, cfg, device)

                st.header("📊 Results")

                # Vuln probability
                col1, col2, col3 = st.columns(3)
                vuln_prob = result["vuln_probability"]
                fixed_prob = result["fixed_probability"]
                verdict = result["vuln_prediction"]

                with col1:
                    color = "#e74c3c" if verdict == "VULNERABLE" else "#2ecc71"
                    st.markdown(
                        f'<div style="text-align:center;padding:20px;background:{color};border-radius:10px;color:white">'
                        f'<h2>{verdict}</h2>'
                        f'<p>Vuln: {vuln_prob:.1%} | Fixed: {fixed_prob:.1%}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.progress(vuln_prob, text=f"Vulnerability probability: {vuln_prob:.1%}")

                with col2:
                    cwe = result["cwe_id"]
                    cwe_name = result["cwe_name"]
                    cwe_conf = result.get("cwe_confidence", 0)
                    st.metric("Predicted CWE", cwe, delta=None)
                    st.caption(cwe_name)
                    if cwe_conf:
                        st.progress(float(cwe_conf), text=f"Confidence: {cwe_conf:.1%}")

                with col3:
                    sev = result["severity"]
                    sev_color = SEVERITY_COLORS.get(sev, "#666")
                    st.markdown(
                        f'<div style="text-align:center;padding:15px;background:{sev_color};border-radius:10px;color:white">'
                        f'<h3>Severity: {sev}</h3>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    sev_probs = result.get("severity_probabilities", {})
                    for s, p in sev_probs.items():
                        st.progress(float(p), text=f"{s}: {p:.1%}")

                # Highlighted code
                st.header("🔎 Suspicious Lines")
                st.caption(f"Top-{top_k} lines highlighted by attribution score (red = more suspicious)")
                source_lines = result.get("source_lines", code_input.splitlines())
                line_scores = result.get("line_scores", {})
                html = highlight_code_html(source_lines, line_scores, top_k)
                st.markdown(html, unsafe_allow_html=True)

                # Code review comment
                st.header("💬 Code Review Comment")
                top_lines = sorted(line_scores.items(), key=lambda x: -x[1])[:top_k] if line_scores else []
                top_line_nums = [i+1 for i, _ in top_lines]
                if verdict == "VULNERABLE":
                    comment = (
                        f"**⚠️ Potential vulnerability detected** ({cwe}: {cwe_name}).\n\n"
                        f"Severity: **{sev}**. "
                    )
                    if top_line_nums:
                        comment += f"The most suspicious lines are: **{', '.join(map(str, top_line_nums))}**. "
                    comment += "\n\nPlease review these lines carefully and ensure proper bounds checking, null pointer handling, or input validation as appropriate."
                else:
                    comment = "✅ No vulnerability pattern detected. This appears to be a fixed/safe version."
                st.info(comment)

            except Exception as e:
                st.error(f"Inference error: {e}")
                logger.error("Inference error", exc_info=True)

    elif model is not None and not code_input.strip():
        st.info("Paste some code or choose a built-in example, then click **Analyze**.")


if __name__ == "__main__":
    main()
