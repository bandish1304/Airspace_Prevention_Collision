# Week 3 Imbalance Strategy Selection

The selection prioritizes safety-critical detection in this order:
false negative rate (lower), recall (higher), F1 (higher), precision (higher), average precision (higher), and AUROC (higher).

Selected strategy: **smote**

The selected strategy will carry forward to the Week 4 graph and later model experiments.

| Strategy | Precision | Recall | F1 | AUROC | Average precision | False negative rate | False positive rate |

| baseline | 0.0000 | 0.0000 | 0.0000 | 0.9994 | 0.8875 | 1.0000 | 0.0000 |
| smote | 0.2105 | 1.0000 | 0.3478 | 0.9994 | 0.8875 | 0.0000 | 0.0179 |
| class_weight | 0.1905 | 1.0000 | 0.3200 | 0.9994 | 0.8875 | 0.0000 | 0.0202 |
| focal_loss | 0.0000 | 0.0000 | 0.0000 | 0.9935 | 0.3046 | 1.0000 | 0.0000 |
