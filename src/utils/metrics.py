from typing import Optional

import numpy as np
from sklearn.metrics import (
    completeness_score,
    homogeneity_score,
    mean_absolute_error,
    mean_squared_error,
    v_measure_score,
)


def rmse(true: np.ndarray, pred: Optional[np.ndarray] = None) -> float:
    if pred is None:
        pred = 0.0 * true
    diff = true.flatten() - pred.flatten()
    return np.sqrt(np.nanmean(pow(diff, 2)))


def mae(true: np.ndarray, pred: Optional[np.ndarray] = None) -> float:
    if pred is None:
        pred = 0.0 * true
    diff = true.flatten() - pred.flatten()
    return np.nanmean(np.abs(diff))


def eval_metrics_reg(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": np.sqrt(mean_squared_error(true, pred)),
        "mae": mean_absolute_error(true, pred),
    }


def eval_metrics_seg(true: list[int], pred: np.ndarray) -> dict[str, float]:
    return {
        "homogeneity": homogeneity_score(true, pred),
        "completeness": completeness_score(true, pred),
        "v_measure": v_measure_score(true, pred),
    }
