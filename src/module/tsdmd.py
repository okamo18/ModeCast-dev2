import random
import time
from typing import Tuple, Optional

import numpy as np

from src.module.nlds import NLDS
from src.module.regime import Regime
from src.module.regime_storage import RegimeStorage
from src.utils.mdb import MDB
from src.utils.metrics import rmse

random.seed(0)


class TSDMD:
    def __init__(
        self,
        n: int,
        d: int,
        h: int,
        k: int,
        w: int,
        lcurr: int,
        lstep: int,
        lrprt: int,
        err_th: float,
        trunc_th: float,
        max_iter: int,
        use_control: bool = False,
        rank: Optional[int] = None,

    ) -> None:
        self.n = n
        self.d = d
        self.h = h
        self.k = k
        self.w = w
        self.lcurr = lcurr
        self.lstep = lstep
        self.lrprt = lrprt
        self.err_th = err_th
        self.trunc_th = trunc_th
        self.max_iter = max_iter

        self.use_control = use_control
        self.rank = rank



    def initialize(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Upsilon: np.ndarray = None,   # ★ 追加: DMDc 用の入力ウィンドウ列
    ) -> RegimeStorage:
        """
        X       : (dxh, N)   状態の遅延埋め込み行列
        Y       : (dxh, N)   1 ステップ先の状態
        Upsilon : (duxh, N)  入力の遅延埋め込み行列（DMDc 用、DMD のときは None でOK）
        """
        init_points = random.sample(
            list(range(X.shape[1] - self.lcurr + self.h - 1)),
            k=self.k,
        )
        regimes: list[Regime] = []

        for i in range(self.k):
            st = init_points[i]
            en = init_points[i] + self.lcurr - self.h + 1

            X_seg = X[:, st:en]
            Y_seg = Y[:, st:en]

            if self.use_control:
                # ★ DMDc モード：入力も一緒に渡す
                assert Upsilon is not None, "TSDMD(use_control=True) のときは Upsilon が必須です"
                U_seg = Upsilon[:, st:en]

                regime = Regime(
                    trunc_th=self.trunc_th,
                    idx=i,
                    use_control=True,
                    rank=self.rank,
                )
                regime.fit(X_seg, Y_seg, U_seg)
            else:
                # ★ これまで通りの DMD モード
                regime = Regime(
                    trunc_th=self.trunc_th,
                    idx=i,
                    use_control=False,
                )
                regime.fit(X_seg, Y_seg)

            regime.update_intervals(st, en)
            regimes.append(regime)

        return RegimeStorage(regimes=regimes)





    def forecast(
        self,
        Xc: np.ndarray,
        regime_storage: RegimeStorage,
        mdb: MDB,
        Upsilon_c: Optional[np.ndarray] = None,
        Upsilon_future: Optional[np.ndarray] = None,
    ) -> Tuple[RegimeStorage, MDB]:
        """
        Xc           : (dxh, n_c)   現在の状態ウィンドウ列
        Upsilon_c    : (duxh, n_c)  現在区間の入力ウィンドウ列（DMDc の新レジーム学習などに使える）
        Upsilon_future : (duxh, L-1) 将来の入力ウィンドウ列（予測に使う, L = lstep+lrprt-1）
        """
        n = Xc.shape[1]

        if not hasattr(self, "rgm_c_idx"):
            self.rgm_c_idx = 0

        _time = time.monotonic()

        # -----------------------------
        # 1. 今のレジーム or 既存レジームの中から最適なものを選ぶ
        #    → RegimeStorage 側は DMD 時代と同じインターフェースを維持
        #       （ここでは u はまだ使わない：Regime.predict が u=0 でフォールバック）
        # -----------------------------
        cand_regime, Vc, err = regime_storage.get_best_regime(
            Xc=Xc,
            rgm_c_idx=self.rgm_c_idx,
            err_th=self.err_th,
        )

        # -----------------------------
        # 2. 誤差が大きければ新しいレジームを作る
        # -----------------------------
        if err > self.err_th:
            # ★ self.use_control に応じて DMD / DMDc どちらのレジームを作るか分岐
            if self.use_control and (Upsilon_c is not None):
                # DMDc レジームを新規作成
                X_seg = Xc[:, :-1]          # (dxh, n_c-1)
                Y_seg = Xc[:, 1:]           # (dxh, n_c-1)
                U_seg = Upsilon_c[:, :-1]   # (duxh, n_c-1)

                new_rgm = Regime(
                    trunc_th=self.trunc_th,
                    idx=len(regime_storage),
                    use_control=True,
                    rank=self.rank,
                )
                new_rgm.fit(X_seg, Y_seg, U_seg)
            else:
                # これまで通り DMD レジームを作る
                new_rgm = Regime(
                    trunc_th=self.trunc_th,
                    idx=len(regime_storage),
                    use_control=False,
                )
                new_rgm.fit(Xc[:, :-1], Xc[:, 1:])

            # NLDS で「そのレジームでどれくらい再現できるか」を評価
            nlds = NLDS(Xc)
            nlds.fit_X0(new_rgm)
            new_Vc = nlds.generate(new_rgm, n)
            new_err = rmse(Xc, new_Vc)

            if new_err < self.err_th:
                regime_storage.append(new_rgm)
                self.rgm_c_idx = new_rgm.idx
                Vc = new_Vc
                print(f"   ERRc > ERRnew ({err:.3f} vs {new_err:.3f})")
                print(f"   ===> new regime{new_rgm.idx} is adopted")
                err = new_err
            else:
                # 既存の候補レジームを使う
                self.rgm_c_idx = cand_regime.idx

        # -----------------------------
        # 3. 誤差が許容範囲なら、既存レジームをオンライン更新
        # -----------------------------
        else:
            self.rgm_c_idx = cand_regime.idx
            new_Xc = Xc[:, -(self.lrprt + 1) :]

            if not self.use_control:
                # もともとの DMD のオンライン学習
                for xt, yt in zip(new_Xc[:, :-1].T, new_Xc[:, 1:].T):
                    cand_regime.learn_one(xt, yt)
            else:
                # DMDc 用のオンライン更新はまだ実装していないので何もしない
                # （やりたくなったらここで ut も一緒に渡すように拡張）
                pass

        # -----------------------------
        # 4. 将来 lstep + lrprt - 1 ステップ分を予測
        #    → ここで DMDc の u をちゃんと使えるようにする
        # -----------------------------
        length = self.lstep + self.lrprt - 1

        if self.use_control:
            # DMDc: Upsilon_future を渡せば u をきちんと使う。
            #       main.py 側で Upsilon_future を作って渡してあげるのが次のステップ。
            Vf = np.real(
                cand_regime.predict(
                    Vc[:, -1],          # 最後のウィンドウ
                    length,             # 出力列数
                    Upsilon_seq=Upsilon_future,  # None の場合は Regime 側で u=0 フォールバック
                    with_initial=True,
                )
            )
        else:
            # DMD: これまで通り
            Vf = np.real(
                cand_regime.predict(
                    Vc[:, -1],
                    length,
                    with_initial=True,
                )
            )

        # 既存の発散防止ロジック
        cond = np.abs(Vf).max(axis=1) > 1.8 * np.abs(Xc).max()
        Vf[cond] = Xc[cond, -(self.lstep + self.lrprt - 1) :]

        # RegimeStorage を更新
        regime_storage.update(cand_regime)
        clock_time = time.monotonic() - _time

        # logging / MDB 更新はそのまま
        mdb.results["time"][mdb.tf : mdb.te] = clock_time
        mdb.results["rgm_idx"][mdb.tm : mdb.tc] = self.rgm_c_idx
        mdb.results["n_rgm"][mdb.tf : mdb.te] = len(regime_storage)
        mdb.logging_seq(Xc, Vc, Vf)

        return regime_storage, mdb

