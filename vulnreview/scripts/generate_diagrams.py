"""
generate_diagrams.py - Generate high-quality architecture and pipeline diagrams
for the IEEE research paper on VulnReview.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_pipeline_diagram(out_path="docs/pipeline.png"):
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Colors
    bg_box = "#F0F4F8"
    border_box = "#1E3A8A"
    accent_box = "#DBEAFE"
    hl_color = "#2563EB"
    text_color = "#1E293B"

    # Title
    ax.text(7, 6.6, "VulnReview: End-to-End System & Experimental Pipeline", 
            ha="center", va="center", fontsize=15, fontweight="bold", color=border_box)

    boxes = [
        {"x": 0.5, "y": 3.8, "w": 2.2, "h": 2.2, "title": "1. Raw Dataset", 
         "lines": ["Big-Vul Benchmark", "4,432 CVE records", "10,501 C/C++ files", "27,603 hunks"]},
        
        {"x": 3.2, "y": 3.8, "w": 2.2, "h": 2.2, "title": "2. Patch-Pair Extraction", 
         "lines": ["Filter diff hunks (+/-)", "17,560 valid hunks", "Before: removed (-) -> vuln=1", "After: added (+) -> vuln=0", "35,106 total samples"]},

        {"x": 5.9, "y": 3.8, "w": 2.2, "h": 2.2, "title": "3. Leakage-Free Splits", 
         "lines": ["Commit-grouped disjoint", "P1: 80/10/10 stratified", "P2: Temporal (<=2017 / 18-19)", "P3: Cross-project split", "Zero commit/text overlap"]},

        {"x": 8.6, "y": 3.8, "w": 2.2, "h": 2.2, "title": "4. Model Benchmarking", 
         "lines": ["B1: TF-IDF + LogReg", "B2: TF-IDF + Random Forest", "B3: CodeBERT Single-Task", "B4: CodeBERT Multi-Task", "Ablations (-CWE, -Sev)"]},

        {"x": 11.3, "y": 3.8, "w": 2.2, "h": 2.2, "title": "5. Evaluation & Review", 
         "lines": ["Vuln F1 & Pair Acc", "CWE macro-F1 & Sev MAE", "Severity-Weighted Recall", "Bootstrap 95% CI", "Paired comparisons"]},

        # Lower row: XAI and App
        {"x": 3.2, "y": 0.8, "w": 3.5, "h": 2.0, "title": "6. Integrated Gradients Attribution", 
         "lines": ["Captum path interpolation (m=30 steps)", "Token embeddings -> Line attributions", "Loc@1 and Loc@3 localization evaluation", "Pinpoints exact vulnerable statements"]},

        {"x": 7.3, "y": 0.8, "w": 3.5, "h": 2.0, "title": "7. Interactive Code Review Demo", 
         "lines": ["Streamlit GUI (app/app.py) + CLI predict", "Interactive color heatmap of suspicious lines", "Multi-task triage: Vuln prob + CWE + Severity", "Instant review feedback for developers"]}
    ]

    for b in boxes:
        # Box shadow
        rect_bg = patches.FancyBboxPatch((b["x"]+0.04, b["y"]-0.04), b["w"], b["h"],
                                        boxstyle="round,pad=0.1,rounding_size=0.15",
                                        facecolor="#E2E8F0", edgecolor="none")
        ax.add_patch(rect_bg)
        # Main box
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"],
                                      boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor=accent_box, edgecolor=border_box, linewidth=1.5)
        ax.add_patch(rect)
        # Title banner
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"] - 0.28, b["title"],
                ha="center", va="center", fontsize=10.5, fontweight="bold", color=border_box)
        # Divider line
        ax.plot([b["x"]+0.1, b["x"]+b["w"]-0.1], [b["y"]+b["h"]-0.5, b["y"]+b["h"]-0.5],
                color=border_box, linewidth=0.8, alpha=0.5)
        # Content lines
        for i, line in enumerate(b["lines"]):
            ax.text(b["x"] + 0.15, b["y"] + b["h"] - 0.8 - i*0.35, f"• {line}",
                    ha="left", va="center", fontsize=8.5, color=text_color)

    # Arrows top row
    arrow_props = dict(facecolor=hl_color, edgecolor=hl_color, width=2.5, headwidth=8, headlength=8)
    for i in range(4):
        x_start = boxes[i]["x"] + boxes[i]["w"] + 0.05
        y_pos = boxes[i]["y"] + boxes[i]["h"]/2
        ax.annotate("", xy=(boxes[i+1]["x"] - 0.05, y_pos), xytext=(x_start, y_pos),
                    arrowprops=arrow_props)

    # Arrow down from box 4 to box 5/6
    ax.annotate("", xy=(5.0, 2.9), xytext=(9.7, 3.7),
                arrowprops=dict(facecolor=hl_color, edgecolor=hl_color, width=2.0, headwidth=7, headlength=7,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate("", xy=(9.0, 2.9), xytext=(12.4, 3.7),
                arrowprops=dict(facecolor=hl_color, edgecolor=hl_color, width=2.0, headwidth=7, headlength=7,
                                connectionstyle="arc3,rad=-0.15"))
    ax.annotate("", xy=(7.2, 1.8), xytext=(6.8, 1.8),
                arrowprops=arrow_props)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


def generate_architecture_diagram(out_path="docs/architecture.png"):
    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Colors
    c_input = "#E0F2FE"       # Light sky
    c_encoder = "#EDE9FE"     # Light purple
    c_pool = "#FEF3C7"        # Amber
    c_vuln = "#FEE2E2"        # Reddish
    c_cwe = "#E0E7FF"         # Indigo
    c_sev = "#DCFCE7"         # Green
    c_loss = "#F1F5F9"        # Slate
    border = "#1E293B"

    # Title
    ax.text(6.5, 7.6, "VulnReview (B4): Multi-Task Patch-Pair CodeBERT Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")

    # 1. Input Box
    box_input = patches.FancyBboxPatch((0.5, 3.2), 2.2, 2.4, boxstyle="round,pad=0.1",
                                      facecolor=c_input, edgecolor="#0284C7", linewidth=2)
    ax.add_patch(box_input)
    ax.text(1.6, 5.2, "Input Code Snippet / Diff", ha="center", va="center", fontweight="bold", fontsize=10, color="#0369A1")
    ax.text(1.6, 4.4, "Diff Hunk / Function\n(Context + Changes)\nMax Tokens: 512\n[CLS] ... [SEP]",
            ha="center", va="center", fontsize=8.5, color="#0F172A")

    # Arrow to Tokenizer & Embedding
    ax.annotate("", xy=(3.4, 4.4), xytext=(2.8, 4.4),
                arrowprops=dict(facecolor="#0284C7", edgecolor="#0284C7", width=2, headwidth=7))

    # 2. CodeBERT Encoder
    box_enc = patches.FancyBboxPatch((3.5, 2.5), 2.8, 3.8, boxstyle="round,pad=0.1",
                                    facecolor=c_encoder, edgecolor="#7C3AED", linewidth=2)
    ax.add_patch(box_enc)
    ax.text(4.9, 6.0, "Pre-trained CodeBERT", ha="center", va="center", fontweight="bold", fontsize=11, color="#6D28D9")
    ax.text(4.9, 5.5, "(microsoft/codebert-base)", ha="center", va="center", fontstyle="italic", fontsize=8.5, color="#4C1D95")
    ax.text(4.9, 4.2, "12-Layer Transformer\n768 Hidden Dim\n12 Attention Heads\n125M Parameters\nBidirectional Self-Attention",
            ha="center", va="center", fontsize=9, color="#1E1B4B")
    ax.text(4.9, 2.8, "Captum Integrated Gradients\nLine Attribution Hooked Here",
            ha="center", va="center", fontsize=8, fontweight="bold", color="#7C3AED")

    # Arrow to Representation
    ax.annotate("", xy=(7.0, 4.4), xytext=(6.4, 4.4),
                arrowprops=dict(facecolor="#7C3AED", edgecolor="#7C3AED", width=2, headwidth=7))

    # 3. Pooling & Dropout
    box_pool = patches.FancyBboxPatch((7.1, 3.5), 1.8, 1.8, boxstyle="round,pad=0.1",
                                     facecolor=c_pool, edgecolor="#D97706", linewidth=2)
    ax.add_patch(box_pool)
    ax.text(8.0, 4.9, "Representation", ha="center", va="center", fontweight="bold", fontsize=10, color="#B45309")
    ax.text(8.0, 4.1, "[CLS] Vector\n(h in R^768)\n+\nDropout (p=0.1)",
            ha="center", va="center", fontsize=8.5, color="#78350F")

    # Arrows from Pooling to 3 Heads
    ax.annotate("", xy=(9.6, 6.2), xytext=(9.0, 4.8),
                arrowprops=dict(facecolor="#D97706", edgecolor="#D97706", width=2, headwidth=6, connectionstyle="arc3,rad=-0.1"))
    ax.annotate("", xy=(9.6, 4.4), xytext=(9.0, 4.4),
                arrowprops=dict(facecolor="#D97706", edgecolor="#D97706", width=2, headwidth=6))
    ax.annotate("", xy=(9.6, 2.6), xytext=(9.0, 4.0),
                arrowprops=dict(facecolor="#D97706", edgecolor="#D97706", width=2, headwidth=6, connectionstyle="arc3,rad=0.1"))

    # 4. Head 1: Vulnerability Head
    box_h1 = patches.FancyBboxPatch((9.7, 5.4), 2.9, 1.6, boxstyle="round,pad=0.1",
                                   facecolor=c_vuln, edgecolor="#DC2626", linewidth=2)
    ax.add_patch(box_h1)
    ax.text(11.15, 6.6, "Head 1: Vulnerability Detection", ha="center", va="center", fontweight="bold", fontsize=9.5, color="#B91C1C")
    ax.text(11.15, 5.9, "Linear (768 -> 2)\nBinary: Vulnerable vs Fixed\nSoftmax Probabilities",
            ha="center", va="center", fontsize=8.5, color="#7F1D1D")

    # Head 2: CWE Classification
    box_h2 = patches.FancyBboxPatch((9.7, 3.6), 2.9, 1.6, boxstyle="round,pad=0.1",
                                   facecolor=c_cwe, edgecolor="#4F46E5", linewidth=2)
    ax.add_patch(box_h2)
    ax.text(11.15, 4.8, "Head 2: CWE Classification", ha="center", va="center", fontweight="bold", fontsize=9.5, color="#3730A3")
    ax.text(11.15, 4.1, "Linear (768 -> 11 classes)\nTop-10 CWEs + 'Other'\nMasked on fixed samples",
            ha="center", va="center", fontsize=8.5, color="#1E1B4B")

    # Head 3: Severity Prediction
    box_h3 = patches.FancyBboxPatch((9.7, 1.8), 2.9, 1.6, boxstyle="round,pad=0.1",
                                   facecolor=c_sev, edgecolor="#16A34A", linewidth=2)
    ax.add_patch(box_h3)
    ax.text(11.15, 3.0, "Head 3: Severity Band", ha="center", va="center", fontweight="bold", fontsize=9.5, color="#15803D")
    ax.text(11.15, 2.3, "Linear (768 -> 3 classes)\nLow (<4.0) | Med (4-7) | High (>=7)\nMasked on fixed samples",
            ha="center", va="center", fontsize=8.5, color="#14532D")

    # 5. Multi-task Loss Box at bottom
    box_loss = patches.FancyBboxPatch((1.5, 0.4), 9.8, 1.0, boxstyle="round,pad=0.1",
                                     facecolor=c_loss, edgecolor="#475569", linewidth=1.5)
    ax.add_patch(box_loss)
    ax.text(6.4, 0.9, r"Joint Multi-Task Loss:   $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{vuln}} + \mathbb{I}[y_{\mathrm{vuln}}=1] \cdot \left(0.5 \cdot \mathcal{L}_{\mathrm{cwe}} + 0.5 \cdot \mathcal{L}_{\mathrm{sev}}\right)$",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0F172A")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_pipeline_diagram()
    generate_architecture_diagram()
