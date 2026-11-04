# ==============================================================================
from dataclasses import dataclass

import numpy as np
import pandas as pd
# ==============================================================================

PLUTCHIK_EMOTIONS  =  [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]

# ==============================================================================

@dataclass(slots  =  True)
class AgreementMetrics:
    observed_agreement:     float
    chance_agreement:       float
    cohens_kappa:           float
    confusion_matrix:       pd.DataFrame
    per_class_agreement:    dict[str, float]

# ==============================================================================

class BenchmarkEvaluator:
    def __init__(
            self,
            labels: list[str] | None  =  None
        ):
        self.labels  =  labels or PLUTCHIK_EMOTIONS

    def compute_agreement(
            self,
            annotator_a: list[str],
            annotator_b: list[str],
        ) -> AgreementMetrics:
        if len(annotator_a) != len(annotator_b):
            raise ValueError("Annotator lists must have identical lengths.")

        n_samples  =  len(annotator_a)
        if n_samples == 0:
            raise ValueError("Cannot compute agreement on empty samples.")

        labels       =  self.labels
        n_classes    =  len(labels)
        label_to_idx =  {l: i for i, l in enumerate(labels)}

        conf_mat  =  np.zeros((n_classes, n_classes), dtype  =  np.int32)
        for a, b in zip(annotator_a, annotator_b):
            if a in label_to_idx and b in label_to_idx:
                conf_mat[label_to_idx[a], label_to_idx[b]]  +=  1

        total_valid  =  int(conf_mat.sum())
        if total_valid == 0:
            raise ValueError("No valid overlapping labels found.")

        p_o  =  float(np.trace(conf_mat)) / total_valid

        row_sums  =  conf_mat.sum(axis  =  1).astype(np.float64)
        col_sums  =  conf_mat.sum(axis  =  0).astype(np.float64)
        p_e       =  float(np.sum(row_sums * col_sums)) / (total_valid ** 2)

        if p_e >= 1.0:
            kappa  =  1.0
        else:
            kappa  =  (p_o - p_e) / (1.0 - p_e)

        per_class: dict[str, float]  =  {}
        for idx, label in enumerate(labels):
            denom  =  row_sums[idx] + col_sums[idx]
            if denom > 0:
                per_class[label]  =  round(2.0 * conf_mat[idx, idx] / denom, 4)
            else:
                per_class[label]  =  0.0

        conf_df  =  pd.DataFrame(
            conf_mat,
            index    =  labels,
            columns  =  labels,
        )

        return AgreementMetrics(
            observed_agreement   =  round(p_o, 4),
            chance_agreement     =  round(p_e, 4),
            cohens_kappa         =  round(kappa, 4),
            confusion_matrix     =  conf_df,
            per_class_agreement  =  per_class,
        )

# ==============================================================================
