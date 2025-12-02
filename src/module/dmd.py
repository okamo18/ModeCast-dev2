from typing import Tuple

import numpy as np


class DMD:
    def __init__(self, trunc_th: float = 0.99, rho: float = 0.999) -> None:
        self.trunc_th = trunc_th
        self.rho = rho

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        Uh, S, V = self.__svd(X)
        self.P_tilde = np.linalg.inv((Uh @ X) @ (Uh @ X).T)
        self.A_tilde = Uh @ Y @ V @ np.linalg.inv(S)

    def learn_one(self, xt: np.ndarray, yt: np.ndarray) -> None:
        xt_tilde = self.__hermite(self.U) @ xt
        yt_tilde = self.__hermite(self.U) @ yt
        self.P_tilde = self.P_tilde / self.rho
        gamma = 1 / (1 + xt_tilde.T @ self.P_tilde @ xt_tilde)
        self.P_tilde = (self.P_tilde - gamma * self.P_tilde @ np.outer(xt_tilde, xt_tilde) @ self.P_tilde) / self.rho
        self.A_tilde += gamma * np.outer(yt_tilde - self.A_tilde @ xt_tilde, xt_tilde) @ self.P_tilde

    def predict(self, X0: np.ndarray, n: int, with_initial: bool = False) -> np.ndarray:
        pred = np.zeros((X0.shape[0], n), dtype=np.complex128)
        Lamb, W = np.linalg.eig(self.A_tilde)
        Phi = self.U @ W
        en = pred.shape[1]
        b = np.linalg.pinv(Phi) @ X0
        for i in range(en):
            pred[:, i] = Phi @ np.diag(Lamb ** (i + 1)) @ b
        if with_initial:
            pred = np.hstack((X0[:, np.newaxis], pred))[:, :-1]

        return np.real(pred)

    def __svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        k = self.__get_k(S)
        self.U = U[:, :k]
        self.S = np.diag(S[:k])
        self.V = self.__hermite(Vh[:k, :])

        return self.__hermite(self.U), self.S, self.V

    def __get_k(self, S: np.ndarray, min_dim: int = 2) -> int:
        return max(int(np.argmax(np.cumsum(S) / np.sum(S) >= self.trunc_th) + 1), min_dim)

    def __hermite(self, X: np.ndarray) -> np.ndarray:
        return X.conj().T
