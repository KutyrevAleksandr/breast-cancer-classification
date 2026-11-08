from __future__ import annotations

import numpy as np


def positive_scores(model, X) -> np.ndarray:
    """Оценка «злокачественности» для ROC-AUC: вероятность, если модель
    её умеет, иначе расстояние до разделяющей поверхности (SVC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
