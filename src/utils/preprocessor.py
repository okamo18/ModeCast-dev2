from typing import Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def create_variables(data: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    d, n = data.shape
    shaped_data = np.empty((h * d, n - h + 1), dtype=np.float64)
    for i in range(h, n + 1):
        shaped_data[:, i - h] = data[:, i - h : i].flatten()
    return shaped_data[:, :-1], shaped_data[:, 1:]


@njit(cache=True)
def create_variables_dmdc(
    state: np.ndarray, control: np.ndarray, h: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DMDc 用の変数生成関数

    Parameters
    ----------
    state : (d_x, n)
        状態系列 x_k
    control : (d_u, n)
        入力系列 u_k
    h : int
        遅延ウィンドウ長

    Returns
    -------
    X      : (h*d_x, n-h)
        過去状態ウィンドウ
    X_next : (h*d_x, n-h)
        1 ステップ先状態ウィンドウ
    Upsilon: (h*d_u, n-h)
        入力ウィンドウ
    """
    # 1D が来てもいいように 2D に揃える
    if state.ndim == 1:
        state = state.reshape(1, -1)
    if control.ndim == 1:
        control = control.reshape(1, -1)

    d_x, n_x = state.shape
    d_u, n_u = control.shape

    if n_x != n_u:
        # numba 内なのでメッセージはシンプルに
        raise ValueError("state and control must have the same length.")

    n = n_x
    m = n - h + 1  # ウィンドウ数

    shaped_state = np.empty((h * d_x, m), dtype=np.float64)
    shaped_control = np.empty((h * d_u, m), dtype=np.float64)

    for i in range(h, n + 1):
        idx = i - h
        shaped_state[:, idx] = state[:, i - h : i].flatten()
        shaped_control[:, idx] = control[:, i - h : i].flatten()

    X = shaped_state[:, :-1]
    X_next = shaped_state[:, 1:]
    Upsilon = shaped_control[:, :-1]

    return X, X_next, Upsilon


def standardize(X: np.ndarray) -> np.ndarray:
    mean = np.nanmean(X, 0)
    std = np.nanstd(X, 0)

    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0  # とりあえず 1 で割ることにする

    X_std = (X - mean) / std_safe

    # 「そもそも定数だった列」は 0 だけにしておくと気持ちよい
    X_std[:, std == 0] = 0.0

    return X_std
