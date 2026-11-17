# VulnReview — Results Summary

> **ALL NUMBERS ARE FROM ACTUAL RUNS. NO VALUES ARE FABRICATED.**

## Table V: Comparison on P1 (Random Split)

| model          | test_f1_str     | test_accuracy_str   | test_precision_str   | test_recall_str   | test_pair_accuracy_str   |
|:---------------|:----------------|:--------------------|:---------------------|:------------------|:-------------------------|
| B1_TFIDF_LR    | 0.5444          | 0.5313              | 0.5296               | 0.5601            | 0.5732                   |
| B2_TFIDF_RF    | 0.5227          | 0.5354              | 0.5373               | 0.5088            | 0.5100                   |
| B3_single_task | 0.6168 ± 0.0065 | 0.5308 ± 0.0051     | 0.5214 ± 0.0045      | 0.7555 ± 0.0282   | 0.6085 ± 0.0204          |
| B4_full        | 0.5432 ± 0.0294 | 0.5165 ± 0.0141     | 0.5143 ± 0.0120      | 0.5766 ± 0.0512   | 0.5636 ± 0.0145          |
| B4_no_cwe      | 0.6018 ± 0.0251 | 0.5140 ± 0.0027     | 0.5098 ± 0.0024      | 0.7378 ± 0.0791   | 0.5918 ± 0.0299          |
| B4_no_sev      | 0.5666 ± 0.0358 | 0.5224 ± 0.0042     | 0.5184 ± 0.0013      | 0.6287 ± 0.0823   | 0.5745 ± 0.0138          |

## Table VI: B4 Generalization Across Protocols

| protocol   | test_f1_str     | test_pair_accuracy_str   | test_cwe_macro_f1_str   |
|:-----------|:----------------|:-------------------------|:------------------------|
| P1         | 0.5432 ± 0.0294 | 0.5636 ± 0.0145          | 0.1430 ± 0.0131         |
| P2         | 0.5942 ± 0.0384 | 0.6139 ± 0.0680          | 0.0948 ± 0.0220         |
| P3         | 0.5819 ± 0.0285 | 0.5554 ± 0.0232          | 0.0815 ± 0.0100         |

## Table VII: Ablation Study (P1)

| model          | test_f1_str     | test_pair_accuracy_str   | test_sw_recall_str   |
|:---------------|:----------------|:-------------------------|:---------------------|
| B3_single_task | 0.6168 ± 0.0065 | 0.6085 ± 0.0204          | 0.7430 ± 0.0232      |
| B4_full        | 0.5432 ± 0.0294 | 0.5636 ± 0.0145          | 0.5509 ± 0.0366      |
| B4_no_cwe      | 0.6018 ± 0.0251 | 0.5918 ± 0.0299          | 0.7257 ± 0.0829      |
| B4_no_sev      | 0.5666 ± 0.0358 | 0.5745 ± 0.0138          | 0.6121 ± 0.0722      |

## Hyperparameters Used

- Model: `microsoft/codebert-base`
- Max length: 512
- Batch size: 16
- Grad accum steps: 2
- Learning rate: 2e-05
- Weight decay: 0.01
- Epochs: 5
- Warmup ratio: 0.1
- fp16: True
- Seeds: [0, 1, 2]

## Explanation Metrics (Integrated Gradients)

- Protocol: P1
- Loc@1: 0.5067
- Loc@3: 0.8167
- Samples explained: 300

## Paired Bootstrap CI: B4 vs B3 (P1)

- Observed F1 difference (B4 - B3): -0.0736
- 95% CI: [-0.1142, -0.0464]
- B4 F1 values: ['0.5550', '0.5648', '0.5097']
- B3 F1 values: ['0.6152', '0.6112', '0.6239']
