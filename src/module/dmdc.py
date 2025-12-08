import numpy as np


class DMDc:
    """
    Dynamic Mode Decomposition with control (DMDc) の実装。

    X_next ≈ A X + B Upsilon
    を満たす A, B を SVD ベースで推定し、
    さらに POD 基底 U_hat に射影した低次元モデル
        x̃_{k+1} = Ã x̃_k + B̃ u_k
    を内部に持つ。
    """

    def __init__(self, rank: int = None, trunc_th: float = 0.99):
        """
        Parameters
        ----------
        rank : int or None
            SVD で次元削減するときのランク。
            None の場合は trunc_th に基づいて自動決定する。
            指定した場合、この値を上限としてランクを決める。
        trunc_th : float
            特異値エネルギーの累積寄与率のしきい値（0〜1）。
        """
        self.rank = rank
        self.trunc_th = trunc_th

        # フル次元の行列
        self.A = None  # (dxh, dxh)
        self.B = None  # (dxh, duxh)

        # POD 基底 & 低次元行列
        self.U_hat = None      # (dxh, r_pod)
        self.A_tilde = None    # (r_pod, r_pod)
        self.B_tilde = None    # (r_pod, duxh)

        # 次元情報
        self._dxh = None
        self._duxh = None
        self._r_pod = None

    def fit(self, X: np.ndarray, X_next: np.ndarray, Upsilon: np.ndarray) -> None:
        """
        DMDc モデルを学習する。

        Parameters
        ----------
        X : (dxh, m)
            過去状態ウィンドウ（h ステップ分の遅延埋め込み）
        X_next : (dxh, m)
            1 ステップ先状態ウィンドウ
        Upsilon : (duxh, m)
            入力ウィンドウ（同じ長さ m）

        Notes
        -----
        まず Brunton+ の DMDc に対応した式でフル次元の A, B を推定し，
        その後で X の POD 基底 U_hat を用いて
            Ã = U_hat^* A U_hat
            B̃ = U_hat^* B
        を計算する。
        """
        dxh, m = X.shape
        duxh, m2 = Upsilon.shape
        if m != m2:
            raise ValueError(f"X and Upsilon must have the same time length, got {m} and {m2}")

        self._dxh = dxh
        self._duxh = duxh

        # --------------------------------------------------
        # 1. Brunton 型 DMDc: 拡張行列 Ω = [X; Upsilon] から A, B を推定
        # --------------------------------------------------
        Omega = np.vstack([X, Upsilon])  # (dxh + duxh, m)

        # Ω = Ũ Σ̃ Ṽ^*
        U_tilde, S_tilde, Vh_tilde = np.linalg.svd(Omega, full_matrices=False)

        # ランク r_omega を決定
        if self.rank is not None:
            r_omega = min(self.rank, U_tilde.shape[1])
        else:
            energy = np.cumsum(S_tilde ** 2) / np.sum(S_tilde ** 2)
            r_omega = int(np.searchsorted(energy, self.trunc_th) + 1)

        U_tilde = U_tilde[:, :r_omega]           # (dxh+duxh, r_omega)
        S_r = S_tilde[:r_omega]                  # (r_omega,)
        V_r = Vh_tilde[:r_omega, :].conj().T     # (m, r_omega)

        # Ũ を状態部分 U1 と入力部分 U2 に分割
        U1 = U_tilde[:dxh, :]                    # (dxh, r_omega)
        U2 = U_tilde[dxh:, :]                    # (duxh, r_omega)

        # Σ̃^{-1}
        S_inv = np.diag(1.0 / S_r)               # (r_omega, r_omega)

        # 中間項 Ṽ Σ̃^{-1}
        middle = V_r @ S_inv                     # (m, r_omega)

        # フル次元の A, B
        # A = X_next Ṽ Σ̃^{-1} Ũ1^*
        # B = X_next Ṽ Σ̃^{-1} Ũ2^*
        self.A = X_next @ (middle @ U1.conj().T)   # (dxh, dxh)
        self.B = X_next @ (middle @ U2.conj().T)   # (dxh, duxh)

        # --------------------------------------------------
        # 2. X に SVD をかけて POD 基底 U_hat を作る
        # --------------------------------------------------
        # X ≈ U_x S_x V_x^*
        U_x, S_x, Vh_x = np.linalg.svd(X, full_matrices=False)

        if self.rank is not None:
            r_pod = min(self.rank, U_x.shape[1])
        else:
            energy_x = np.cumsum(S_x ** 2) / np.sum(S_x ** 2)
            r_pod = int(np.searchsorted(energy_x, self.trunc_th) + 1)

        self._r_pod = r_pod
        self.U_hat = U_x[:, :r_pod]              # (dxh, r_pod)

        # --------------------------------------------------
        # 3. 低次元行列 Ã, B̃ を計算
        # --------------------------------------------------
        # Ã = U_hat^* A U_hat
        # B̃ = U_hat^* B
        self.A_tilde = self.U_hat.conj().T @ self.A @ self.U_hat      # (r_pod, r_pod)
        self.B_tilde = self.U_hat.conj().T @ self.B                   # (r_pod, duxh)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        1 ステップ予測

        x_{k+1} = A x_k + B u_k        （フル次元）
        ではなく，
        x̃_{k+1} = Ã x̃_k + B̃ u_k    （低次元）
        x_{k+1} = U_hat x̃_{k+1}       （復元）
        を用いて計算する。
        """
        if self.A_tilde is None or self.B_tilde is None or self.U_hat is None:
            raise RuntimeError("DMDc model is not fitted yet. Call `fit` first.")

        # (dxh,) → (dxh, 1)
        x = x.reshape(self._dxh, -1)
        u = u.reshape(self._duxh, -1)

        # 低次元表現 x̃ = U_hat^* x
        x_tilde = self.U_hat.conj().T @ x          # (r_pod, 1)

        # 低次元で 1 ステップ予測
        x_tilde_next = self.A_tilde @ x_tilde + self.B_tilde @ u   # (r_pod, 1)

        # フル空間に戻す x_next = U_hat x̃_next
        x_next = self.U_hat @ x_tilde_next         # (dxh, 1)

        return x_next.reshape(-1)
