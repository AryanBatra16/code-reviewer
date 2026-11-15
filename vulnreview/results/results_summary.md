# VulnReview — Results Summary

> **ALL NUMBERS ARE FROM ACTUAL RUNS. NO VALUES ARE FABRICATED.**

## Table V: Comparison on P1 (Random Split)

| model          | test_f1_str     | test_accuracy_str   | test_precision_str   | test_recall_str   | test_pair_accuracy_str   |
|:---------------|:----------------|:--------------------|:---------------------|:------------------|:-------------------------|
| B1_TFIDF_LR    | 0.5444          | 0.5313              | 0.5296               | 0.5601            | 0.5732                   |
| B2_TFIDF_RF    | 0.5227          | 0.5354              | 0.5373               | 0.5088            | 0.5100                   |
| B3_single_task | 0.6168 ± 0.0065 | 0.5308 ± 0.0051     | 0.5214 ± 0.0045      | 0.7555 ± 0.0282   | 0.6085 ± 0.0204          |
| B4_full        | 0.6465          | 0.5270              | 0.5161               | 0.8649            | 0.6351                   |

## Table VI: B4 Generalization Across Protocols

| protocol   |   test_f1_str |   test_pair_accuracy_str |   test_cwe_macro_f1_str |
|:-----------|--------------:|-------------------------:|------------------------:|
| P1         |        0.6465 |                   0.6351 |                  0.0839 |

## Table VII: Ablation Study (P1)

| model          | test_f1_str     | test_pair_accuracy_str   | test_sw_recall_str   |
|:---------------|:----------------|:-------------------------|:---------------------|
| B3_single_task | 0.6168 ± 0.0065 | 0.6085 ± 0.0204          | 0.7430 ± 0.0232      |
| B4_full        | 0.6465          | 0.6351                   | 0.8634               |

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
- Loc@1: 0.0000
- Loc@3: 0.6000
- Samples explained: 5

## Paired Bootstrap CI: B4 vs B3 (P1)

*Insufficient data: B4 seeds=1, B3 seeds=3. Need ≥2 each.*
