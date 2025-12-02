import numpy as np


class DMDc:
    """
    Dynamic Mode Decomposition with control (DMDc) の実装。

    X_next ≈ A X + B Upsilon
    を満たす A, B を SVD ベースで推定する。
    """

    def __init__(self, rank: int = None, trunc_th: float = 0.99):
        """
        Parameters
        ----------
        rank : int or None
            SVD で次元削減するときのランク。
            None の場合は trunc_th に基づいて自動決定する。
        trunc_th : float
            特異値エネルギーの累積寄与率のしきい値（0〜1）。
        """
        self.rank = rank
        self.trunc_th = trunc_th
        self.A = None
        self.B = None
        self._dxh = None
        self._duxh = None

    def fit(self, X: np.ndarray, X_next: np.ndarray, Upsilon: np.ndarray) -> None:
        """
        DMDc モデルを学習する。

        Parameters
        ----------
        X : (dxh, m)
        X_next : (dxh, m)
        Upsilon : (duxh, m)

        Notes
        -----
        Brunton+ の DMDc に対応した推定法：

            Ω = [X; Upsilon]
            Ω = Ũ Σ̃ Ṽ*
            Ũ = [Ũ1; Ũ2]

            A = X_next Ṽ Σ̃^{-1} Ũ1*
            B = X_next Ṽ Σ̃^{-1} Ũ2*
        """
        dxh, m = X.shape
        duxh, m2 = Upsilon.shape
        if m != m2:
            raise ValueError(f"X and Upsilon must have the same time length, got {m} and {m2}")

        self._dxh = dxh
        self._duxh = duxh

        # 1. 拡張行列
        Omega = np.vstack([X, Upsilon])  # (dxh + duxh, m)

        # 2. SVD
        U_tilde, S_tilde, Vh_tilde = np.linalg.svd(Omega, full_matrices=False)

        # 3. ランク決定
        if self.rank is not None:
            r = min(self.rank, U_tilde.shape[1])
        else:
            energy = np.cumsum(S_tilde ** 2) / np.sum(S_tilde ** 2)
            r = int(np.searchsorted(energy, self.trunc_th) + 1)

        U_tilde = U_tilde[:, :r]
        S_r = S_tilde[:r]
        V_r = Vh_tilde[:r, :].conj().T  # (m, r)

        # Ũ を分割
        U1 = U_tilde[:dxh, :]   # (dxh, r)
        U2 = U_tilde[dxh:, :]   # (duxh, r)

        # Σ^{-1}
        S_inv = np.diag(1.0 / S_r)

        # 中間項
        middle = V_r @ S_inv  # (m, r)

        # 4. A, B の推定
        self.A = X_next @ (middle @ U1.conj().T)
        self.B = X_next @ (middle @ U2.conj().T)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """1 ステップ予測 x_{k+1} = A x_k + B u_k"""
        if self.A is None or self.B is None:
            raise RuntimeError("DMDc model is not fitted yet. Call `fit` first.")
        x = x.reshape(self._dxh, -1)
        u = u.reshape(self._duxh, -1)
        x_next = self.A @ x + self.B @ u
        return x_next.reshape(-1)
