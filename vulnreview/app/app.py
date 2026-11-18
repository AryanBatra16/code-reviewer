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
    return predict_one(text, model, tokenizer, label_maps, cfg, device, n_ig_steps=50)


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
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: linear-gradient(135deg,#090d14,#101625); }
        [data-testid="stSidebar"] {
            background:#0d1117;
            border-right:1px solid #20293a;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius:8px;padding:8px 10px;margin:3px 0;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background:#20294a;
        }
        [data-testid="stHeader"] { background:rgba(9,13,20,.75); }
        .stTabs [data-baseweb="tab-list"] { gap:8px; }
        .stTabs [data-baseweb="tab"] {
          background:#151b2a;border:1px solid #222a3d;
          border-radius:10px 10px 4px 4px;padding:10px 18px;
        }
        .stTabs [aria-selected="true"] { background:#20294a;border-bottom:3px solid #667eea; }
        [data-testid="stMetric"] { background:#111827;border:1px solid #273244;border-radius:12px;padding:14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 style="margin-bottom:0">🛡️ VulnReview</h1>'
        '<p style="color:#9aa4b2;margin-top:4px">'
        'Patch-Pair Multi-Task AI for Vulnerability-Aware Code Review</p>',
        unsafe_allow_html=True,
    )

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

    with st.sidebar:
        st.markdown("## 🛡️ VulnReview")
        st.caption("AI-powered code security review")
        st.divider()
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔍 Analyze Code", "📊 Training Results", "🧠 Model & Pipeline", "📚 About"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Author**")
        st.caption("Aryan Batra · 24CSU027\n\nThe NorthCap University")

    if page == "🏠 Home":
        st.markdown("## Welcome to VulnReview")
        st.markdown(
            "Use the sidebar to analyze code, review experimental results, inspect the model "
            "pipeline, or learn about the project."
        )
        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("Analyze Code", "Start")
        with h2:
            st.metric("Training Pairs", "35,106")
        with h3:
            st.metric("Loc@3", "81.67%")

    if page == "🔍 Analyze Code":
        if model is None:
            st.error(f"❌ Model checkpoint not found at: `{ckpt_path}`")

        st.markdown("### 📝 Submit Code")
        input_mode = st.radio(
            "Input mode",
            ["Type / paste code", "Upload file"],
            horizontal=True,
        )

        code_input = ""
        if input_mode == "Upload file":
            uploaded = st.file_uploader(
                "Upload a C/C++ file or .diff file",
                type=["c", "cpp", "cc", "h", "diff", "patch"],
            )
            if uploaded:
                code_input = uploaded.read().decode("utf-8", errors="replace")
                st.code(code_input, language="c")
        else:
            code_input = st.text_area(
                "Code input",
                height=250,
                placeholder="void foo() {\n    char buf[8];\n    gets(buf);  // vulnerable!\n}",
            )

        if st.button(
            "🔍 Analyze for Vulnerabilities",
            type="primary",
            disabled=(model is None or not code_input.strip()),
            use_container_width=True,
        ):
            with st.spinner("Running VulnReview with 50-step Integrated Gradients..."):
                try:
                    result = predict(code_input, model, tokenizer, label_maps, cfg, device)
                    st.header("📊 Analysis Results")

                    col1, col2, col3 = st.columns(3)
                    vuln_prob = result["vuln_probability"]
                    verdict = result["vuln_prediction"]
                    cwe = result["cwe_id"]
                    cwe_name = result["cwe_name"]
                    cwe_conf = result.get("cwe_confidence", 0)
                    sev = result["severity"]

                    with col1:
                        color = "#e74c3c" if verdict == "VULNERABLE" else "#2ecc71"
                        st.markdown(
                            f'<div style="text-align:center;padding:20px;background:{color};border-radius:10px;color:white">'
                            f'<h2>{verdict}</h2><p>Vulnerability confidence: {vuln_prob:.1%}</p></div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(vuln_prob, text=f"Vulnerable: {vuln_prob:.1%}")
                    with col2:
                        st.metric("Predicted CWE", cwe)
                        st.caption(cwe_name)
                        if cwe_conf:
                            st.progress(float(cwe_conf), text=f"CWE confidence: {cwe_conf:.1%}")
                    with col3:
                        sev_color = SEVERITY_COLORS.get(sev, "#666")
                        st.markdown(
                            f'<div style="text-align:center;padding:15px;background:{sev_color};border-radius:10px;color:white">'
                            f'<h3>Severity: {sev}</h3></div>',
                            unsafe_allow_html=True,
                        )
                        for name, prob in result.get("severity_probabilities", {}).items():
                            st.progress(float(prob), text=f"{name}: {prob:.1%}")

                    st.header("🔎 Suspicious Lines")
                    source_lines = result.get("source_lines", code_input.splitlines())
                    line_scores = result.get("line_scores", {})
                    top_k = len(source_lines)
                    st.caption("All lines are highlighted using Integrated Gradients attribution")
                    st.markdown(
                        highlight_code_html(source_lines, line_scores, top_k),
                        unsafe_allow_html=True,
                    )

                    st.header("💬 Code Review Comment")
                    top_lines = sorted(line_scores.items(), key=lambda x: -x[1])[:3]
                    top_line_nums = [i + 1 for i, _ in top_lines]
                    if verdict == "VULNERABLE":
                        comment = (
                            f"**⚠️ Potential vulnerability detected** ({cwe}: {cwe_name}). "
                            f"Severity: **{sev}**."
                        )
                        if top_line_nums:
                            comment += f" The most suspicious lines are **{', '.join(map(str, top_line_nums))}**."
                    else:
                        comment = "✅ No vulnerability pattern detected. This appears to be a fixed/safe version."
                    st.info(comment)
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    logger.error("Inference error", exc_info=True)
        elif model is not None and not code_input.strip():
            st.info("Paste code or upload a file, then click **Analyze for Vulnerabilities**.")

    if page == "📊 Training Results":
        st.markdown("## 📊 Training Results")
        st.caption("Results reported from the project evaluation protocols.")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("P1 Test F1", "0.5432")
        with r2:
            st.metric("P2 Test F1", "0.5942")
        with r3:
            st.metric("P3 Test F1", "0.5819")

        st.markdown("### Line Localization")
        l1, l2 = st.columns(2)
        with l1:
            st.metric("Loc@1", "50.67%")
        with l2:
            st.metric("Loc@3", "81.67%")

        if results_dir.exists():
            result_files = sorted(results_dir.glob("train_results_*.csv"))
            if result_files:
                st.markdown("### Available Experiment Results")
                for result_file in result_files:
                    with st.expander(result_file.name):
                        try:
                            st.dataframe(
                                pd.read_csv(result_file).round(4),
                                use_container_width=True,
                                hide_index=True,
                            )
                        except Exception as e:
                            st.warning(f"Could not read results: {e}")

    if page == "🧠 Model & Pipeline":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Base Model", "CodeBERT")
        with c2:
            st.metric("IG Precision", "50 steps")
        with c3:
            st.metric("Test F1", test_f1)
        with c4:
            st.metric("Loc@3", "81.67%")

        st.markdown("### Model Details")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(
                f"- **Checkpoint:** `{Path(ckpt_path).name if ckpt_path else 'Not loaded'}`\n"
                "- **Base:** `microsoft/codebert-base` (125M)\n"
                "- **Training protocol:** P1 (Random, commit-grouped)\n"
                f"- **Device:** `{device}`"
            )
        with d2:
            st.markdown(
                "- **Heads:** Vulnerability, CWE, Severity\n"
                "- **Dataset:** Big-Vul · 35,106 pairs\n"
                "- **Loc@1:** 50.67%\n"
                "- **Loc@3:** 81.67%"
            )

        st.markdown("### Inference Pipeline")
        p1, p2, p3, p4 = st.columns(4)
        for column, title, detail in [
            (p1, "1 · Tokenize", "CodeBERT tokenizer"),
            (p2, "2 · Encode", "12 Transformer layers"),
            (p3, "3 · Predict", "Three task-specific heads"),
            (p4, "4 · Explain", "Integrated Gradients → lines"),
        ]:
            with column:
                st.info(f"**{title}**\n\n{detail}")

    if page == "📚 About":
        st.markdown("### About VulnReview")
        st.markdown(
            "VulnReview is a patch-pair multi-task system for vulnerability-aware code review. "
            "It combines CodeBERT classification with Integrated Gradients line attribution."
        )
        st.markdown(
            "Built on [Big-Vul](https://github.com/ZeoVan/MSR_20_Code_vulnerability_csv_Dataset) "
            "and [CodeBERT](https://arxiv.org/abs/2002.08155).\n\n"
            "**Author:** Aryan Batra (24CSU027), The NorthCap University"
        )


if __name__ == "__main__":
    main()
