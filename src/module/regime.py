from typing import Optional
import numpy as np

from src.module.dmd import DMD
from src.module.dmdc import DMDc


class Regime:
    """
    1つの「レジーム」に対応する線形モデル（DMD or DMDc）を内側に持つラッパー。
    """

    def __init__(
        self,
        trunc_th: float,
        idx: int = 0,
        use_control: bool = False,
        rank: Optional[int] = None,   # ← ここを Optional[int] に
    ) -> None:
        self.intervals: list = []
        self.idx = idx
        self.use_control = use_control

        # 中身として持つモデルを選択
        if use_control:
            # DMDc モデル（今作ったやつ）
            self.model = DMDc(rank=rank)
        else:
            # 既存の DMD モデル
            self.model = DMD(trunc_th=trunc_th)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Regime):
            return self.idx == other.idx
        return False

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Upsilon: Optional[np.ndarray] = None,  # ← ここも Optional
    ) -> None:
        """
        DMD / DMDc どちらでも使えるようにするためのラッパー。

        - DMD      : fit(X, Y)
        - DMDc     : fit(X, Y, Upsilon)
        """
        if self.use_control:
            assert Upsilon is not None, "DMDc を使う場合は Upsilon (入力列) が必須です"
            self.model.fit(X, Y, Upsilon)
        else:
            self.model.fit(X, Y)

    def update_intervals(self, st: int, en: int) -> None:
        add_flag = True
        for interval in self.intervals:
            assert interval[1] < en
            if interval[0] <= st <= interval[1]:
                interval[1] = en
                add_flag = False
        if add_flag:
            self.intervals.append([st, en])

    def predict(
        self,
        X0: np.ndarray,
        length: int,
        Upsilon_seq: Optional[np.ndarray] = None,  # ← Optional
        with_initial: bool = True,
    ) -> np.ndarray:
        """
        length ステップ先までの予測を返す共通ラッパー。
        - DMD   : 既存の predict(X0, length, with_initial)
        - DMDc  : DMDc.step を length 回まわしてシミュレーション
        """
        if not self.use_control:
            # 既存の DMD のインターフェースに丸投げ
            return self.model.predict(X0, length, with_initial=with_initial)

        assert Upsilon_seq is not None, "DMDc の予測には Upsilon_seq が必要です"

        x = X0.copy()
        dxh = X0.shape[0]
        preds = np.zeros((dxh, length), dtype=float)

        for t in range(length):
            u = Upsilon_seq[:, t]
            x = self.model.step(x, u)
            preds[:, t] = x

        if with_initial:
            return np.hstack([X0.reshape(-1, 1), preds])
        else:
            return preds

    def get_modes(self, X0: np.ndarray, length: int) -> np.ndarray:
        """
        DMD / DMDc どちらのときも「モード展開」を返す。

        ひとまず：
        - DMD   : 既存どおり (A_tilde, U) を利用
        - DMDc  : フル行列 A の固有分解からモードを計算（簡易版）
        """
        if not self.use_control:
            Lamb, W = np.linalg.eig(self.model.A_tilde)
            Phi = self.model.U @ W
        else:
            Lamb, W = np.linalg.eig(self.model.A)
            Phi = W

        modes = np.zeros([Lamb.shape[0], length], dtype="complex")
        b = np.linalg.pinv(Phi) @ X0
        for i in range(length):
            modes[:, i] = np.diag(Lamb**i) @ b
        return modes
