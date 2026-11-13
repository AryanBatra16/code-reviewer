# VulnReview Architecture

```text
+-----------------------------------------------------------------------+
|                             Input Code                                |
|             (C/C++ Snippet or Unified Diff Patch Hunk)                |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    Tokenizer (microsoft/codebert-base)                |
|               (Max tokens: 512, with surrounding context)             |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                         CodeBERT Base Encoder                         |
|                 (12 layers, 768 hidden size, 12 heads)                |
+-----------------------------------------------------------------------+
                                    |
                            [CLS] Representation
                                    |
                                    v
+-----------------------------------------------------------------------+
|                         Dropout Layer (p = 0.1)                       |
+-----------------------------------------------------------------------+
         |                              |                              |
         v                              v                              v
+------------------+          +-------------------+          +------------------+
|    Vuln Head     |          |     CWE Head      |          |  Severity Head   |
| Linear(768 -> 2) |          | Linear(768 -> 11) |          | Linear(768 -> 3) |
+------------------+          +-------------------+          +------------------+
         |                              |                              |
         v                              v                              v
   P(Vulnerable)                Top-10 CWE / Other             Low / Med / High
         |
         +------------------------------+
                                        |
                                        v
                    +---------------------------------------+
                    | Integrated Gradients Attribution      |
                    | (Captum: word embeddings attribution) |
                    +---------------------------------------+
                                        |
                                        v
                    +---------------------------------------+
                    | Line-Level Localization (Loc@1, Loc@3)|
                    | Highlighted Code Review Comments UI   |
                    +---------------------------------------+
```

## Loss Formulation
$$\mathcal{L} = \mathcal{L}_{\text{vuln}} + \mathbb{I}[y_{\text{vuln}}=1] \cdot \left(0.5 \cdot \mathcal{L}_{\text{cwe}} + 0.5 \cdot \mathcal{L}_{\text{sev}}\right)$$

- $\mathcal{L}_{\text{vuln}}$: Class-weighted Cross-Entropy loss on vulnerability detection across all samples.
- $\mathcal{L}_{\text{cwe}}$: Class-weighted Cross-Entropy on CWE classification (only for vulnerable before samples, masked with $-100$ on fixed samples).
- $\mathcal{L}_{\text{sev}}$: Class-weighted Cross-Entropy on Severity band classification (only for vulnerable before samples with known CVSS score).

