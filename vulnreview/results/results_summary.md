# VulnReview — Results Summary

> **ALL NUMBERS ARE FROM ACTUAL RUNS. NO VALUES ARE FABRICATED.**

## Table V: Comparison on P1 (Random Split)

| model          |   test_f1_str |   test_accuracy_str |   test_precision_str |   test_recall_str |   test_pair_accuracy_str |
|:---------------|--------------:|--------------------:|---------------------:|------------------:|-------------------------:|
| B1_TFIDF_LR    |        0.5444 |              0.5313 |               0.5296 |            0.5601 |                   0.5732 |
| B2_TFIDF_RF    |        0.5227 |              0.5354 |               0.5373 |            0.5088 |                   0.51   |
| B3_single_task |        0.6152 |              0.5288 |               0.5199 |            0.7534 |                   0.632  |
| B4_full        |        0.6465 |              0.527  |               0.5161 |            0.8649 |                   0.6351 |

## Table VI: B4 Generalization Across Protocols

| protocol   |   test_f1_str |   test_pair_accuracy_str |   test_cwe_macro_f1_str |
|:-----------|--------------:|-------------------------:|------------------------:|
| P1         |        0.6465 |                   0.6351 |                  0.0839 |

## Table VII: Ablation Study (P1)

| model          |   test_f1_str |   test_pair_accuracy_str |   test_sw_recall_str |
|:---------------|--------------:|-------------------------:|---------------------:|
| B3_single_task |        0.6152 |                   0.632  |               0.74   |
| B4_full        |        0.6465 |                   0.6351 |               0.8634 |

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
