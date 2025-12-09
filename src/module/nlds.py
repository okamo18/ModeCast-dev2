import lmfit
import numpy as np
from typing import Optional  # ★ 追加
from src.module.regime import Regime

XTLi = 0.1
FTLi = 0.1
MAXFEVi = 20


class NLDS:
    def __init__(
        self,
        X: np.ndarray,
        Upsilon: Optional[np.ndarray] = None,  # ★ 制御の遅延埋め込みも受け取る
    ) -> None:
        """
        X       : (d*h, n)   状態の遅延埋め込み
        Upsilon : (du*h, n)  制御の遅延埋め込み（DMDc 用 / DMD なら None でOK）
        """
        self.X = X
        self.Upsilon = Upsilon          # ★ あとで generate で使う
        self.X0 = X[:, 0].copy()
        self.dh = X.shape[0]

    def fit_X0(self, model: "Regime", dps: int = 1) -> "NLDS":
        return nl_fit(self, model, dps)

    def generate(self, model: "Regime", n: int) -> np.ndarray:
        """
        DMD / DMDc の両方に対応した系列生成。

        - model.use_control == False（DMD）
            → これまで通り X0 から n ステップ先まで with_initial=True で生成
        - model.use_control == True（DMDc）
            → Upsilon（制御遅延埋め込み）から Upsilon_seq を切り出して渡す
        """
        # -----------------------
        # 1) 制御なし DMD の場合
        # -----------------------
        if not getattr(model, "use_control", False):
            # もともとの実装と同じ
            return model.predict(self.X0, n, with_initial=True)

        # -----------------------
        # 2) 制御あり DMDc の場合
        # -----------------------
        # Regime は self.model に DMDc インスタンスを持っている想定
        # A, B の次元から duxh を決める
        if hasattr(model.model, "B") and model.model.B is not None:
            duxh = model.model.B.shape[1]
        else:
            # 念のためのフォールバック（ほぼ使わない想定）
            duxh = self.dh

        # with_initial=True のとき、Regime.predict は
        #   - 出力列数: n
        #   - 必要な u の本数: n-1
        # なので Upsilon_seq.shape = (duxh, n-1) を目指す
        if self.Upsilon is None:
            # 制御系列が与えられていない場合は、ゼロ入力を仮定
            Upsilon_seq = np.zeros((duxh, max(n - 1, 0)), dtype=float)
        else:
            # self.Upsilon.shape = (duxh, N_c) のはず
            # 足りなければ 0 でパディングする
            N_c = self.Upsilon.shape[1]
            need = max(n - 1, 0)
            if N_c >= need:
                Upsilon_seq = self.Upsilon[:, :need]
            else:
                Upsilon_seq = np.zeros((duxh, need), dtype=float)
                Upsilon_seq[:, :N_c] = self.Upsilon

        # DMDc モデルで「制御つき」で予測
        return model.predict(
            self.X0,
            n,
            Upsilon_seq=Upsilon_seq,
            with_initial=True,
        )



def nl_fit(nlds: "NLDS", model: "Regime", dps: int) -> "NLDS":
    # (1) create param set
    P = _createP(nlds)
    # (2) start lmfit
    lmsol = lmfit.Minimizer(_objective, P, fcn_args=(nlds.X, nlds, model, dps))
    res = lmsol.leastsq(xtol=XTLi, ftol=FTLi, max_nfev=MAXFEVi)
    # (3) update param set
    nlds = _updateP(res.params, nlds)
    return nlds


def _createP(nlds: "NLDS") -> lmfit.parameter.Parameters:
    P = lmfit.Parameters()
    dh = nlds.dh
    V = True
    for i in range(dh):
        P.add("X0_%i" % (i), value=nlds.X0[i], vary=V)
    return P


def _updateP(P: dict, nlds: "NLDS") -> "NLDS":
    dh = nlds.dh
    for i in range(dh):
        nlds.X0[i] = P["X0_%i" % (i)].value
    return nlds


def _objective(P: dict, X: np.ndarray, nlds: "NLDS", model: "Regime", dps: int) -> np.ndarray:
    n = X.shape[1]
    nlds = _updateP(P, nlds)
    # generate seq
    pred = np.real(nlds.generate(model, n))
    if dps > 1:
        X = X[:, range(0, n, dps)]
    # diffs
    diff = X.flatten() - pred.flatten()
    diff[np.isnan(diff)] = 0

    return diff
