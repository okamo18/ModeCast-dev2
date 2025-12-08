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
        rank: Optional[int] = None,
    ) -> None:
        # もともとの Regime が持っていた情報
        self.intervals: list = []
        self.idx = idx
        self.use_control = use_control

        # 中身のモデルを DMD / DMDc から選択
        if use_control:
            # DMDc モデル（縮約 DMDc）
            # → dmdc.DMDc の __init__(rank, trunc_th=...) を想定
            self.model = DMDc(rank=rank, trunc_th=trunc_th)
        else:
            # 既存の DMD モデル
            self.model = DMD(trunc_th=trunc_th)

    # ------------------------------------------------------------------
    #  もともとの Regime と互換な部分
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Regime):
            return self.idx == other.idx
        return False

    def update_intervals(self, st: int, en: int) -> None:
        add_flag = True
        for interval in self.intervals:
            assert interval[1] < en
            if interval[0] <= st <= interval[1]:
                interval[1] = en
                add_flag = False
        if add_flag:
            self.intervals.append([st, en])

    # ------------------------------------------------------------------
    #  共通インターフェース: fit / learn_one / predict / get_modes
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Upsilon: Optional[np.ndarray] = None,
    ) -> None:
        """
        モデルを学習する共通ラッパー。

        - DMD   : fit(X, Y)
        - DMDc  : fit(X, Y, Upsilon)
        """
        if self.use_control:
            assert Upsilon is not None, "Regime(use_control=True) のときは Upsilon が必須です"
            self.model.fit(X, Y, Upsilon)
        else:
            self.model.fit(X, Y)

    def learn_one(
        self,
        xt: np.ndarray,
        yt: np.ndarray,
        ut: Optional[np.ndarray] = None,
    ) -> None:
        """
        1ステップ分のオンライン更新。

        - DMD   : 既存の learn_one をそのまま呼ぶ
        - DMDc  : ひとまず何もしない（TODO: オンライン DMDc を入れたくなったら実装）
        """
        if not self.use_control:
            # 元の DMD のオンライン更新
            self.model.learn_one(xt, yt)
        else:
            # 今は no-op（レジーム推定の枠組みだけ先に動かす）
            return

    def predict(
        self,
        X0: np.ndarray,
        length: int,
        Upsilon_seq: Optional[np.ndarray] = None,
        with_initial: bool = True,
    ) -> np.ndarray:
        """
        length ステップ先までの予測を返す共通ラッパー。

        - DMD   : 既存の predict(X0, length, with_initial)
                  （出力形状は (dxh, length)）
        - DMDc  : DMDc.step を回して、出力形状が (dxh, length) になるように合わせる
        """
        # --- DMD（制御なし）の場合はそのまま既存実装に丸投げ ---
        if not self.use_control:
            return self.model.predict(X0, length, with_initial=with_initial)

        # --- ここから DMDc 用 ---
        dxh = X0.shape[0]

        # 入力次元 duxh を決める
        if hasattr(self.model, "B") and self.model.B is not None:
            duxh = self.model.B.shape[1]
        else:
            duxh = dxh  # 念のためのフォールバック

        # Upsilon_seq が渡されていない場合 → ゼロ入力を仮定
        if Upsilon_seq is None:
            if length <= 1:
                Upsilon_seq = np.zeros((duxh, 0), dtype=float)
            else:
                # 遷移は (length-1) 回なので、そのぶんだけあればよい
                Upsilon_seq = np.zeros((duxh, length - 1), dtype=float)

        # ここからは Upsilon_seq がある前提
        T_u = Upsilon_seq.shape[1]

        if with_initial:
            # 出力形状は (dxh, length) にする
            preds = np.zeros((dxh, length), dtype=float)
            preds[:, 0] = X0
            x = X0.copy()

            # t = 1, ..., length-1 で遷移を回す
            for t in range(1, length):
                if t - 1 < T_u:
                    u = Upsilon_seq[:, t - 1]
                else:
                    u = np.zeros((duxh,), dtype=float)
                x = self.model.step(x, u)
                preds[:, t] = x

            return preds
        else:
            # with_initial=False の場合:
            # 「X0 から length ステップ先まで」の系列を返す
            preds = np.zeros((dxh, length), dtype=float)
            x = X0.copy()

            for t in range(length):
                if t < T_u:
                    u = Upsilon_seq[:, t]
                else:
                    u = np.zeros((duxh,), dtype=float)
                x = self.model.step(x, u)
                preds[:, t] = x

            return preds


    def get_modes(self, X0: np.ndarray, length: int) -> np.ndarray:
        """
        DMD / DMDc どちらのときも「モード展開」を返す。

        ひとまず：
        - DMD   : 既存どおり (A_tilde, U) を利用
        - DMDc  : フル行列 A の固有分解からモードを計算（簡易版）
        """
        if not self.use_control:
            # 既存 DMD と同じ
            Lamb, W = np.linalg.eig(self.model.A_tilde)
            Phi = self.model.U @ W
        else:
            # DMDc: A の固有分解からモードを定義
            Lamb, W = np.linalg.eig(self.model.A)
            Phi = W

        modes = np.zeros([Lamb.shape[0], length], dtype="complex")
        b = np.linalg.pinv(Phi) @ X0
        for i in range(length):
            modes[:, i] = np.diag(Lamb**i) @ b
        return modes
