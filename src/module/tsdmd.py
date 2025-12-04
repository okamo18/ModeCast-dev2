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
        Upsilon_c: np.ndarray = None,   # ★ 追加：制御入力のウィンドウ（あとで使う）
    ) -> Tuple[RegimeStorage, MDB]:
        n = Xc.shape[1]
        if not hasattr(self, 'rgm_c_idx'):
            self.rgm_c_idx = 0
        _time = time.monotonic()
        cand_regime, Vc, err = regime_storage.get_best_regime(
            Xc=Xc,
            rgm_c_idx=self.rgm_c_idx,
            err_th=self.err_th,
            Upsilon_c=Upsilon_c,  # ← 追加
        )
        if err > self.err_th:
            new_rgm = Regime(trunc_th=self.trunc_th, idx=len(regime_storage))
            new_rgm.fit(Xc[:, :-1], Xc[:, 1:])
            nlds = NLDS(Xc)
            nlds.fit_X0(new_rgm)
            new_Vc = nlds.generate(new_rgm, n)
            new_err = rmse(Xc, new_Vc)
            if new_err < self.err_th:
                regime_storage.append(new_rgm)
                self.rgm_c_idx = new_rgm.idx
                Vc = new_Vc
                print(f'   ERRc > ERRnew ({err:.3f} vs {new_err:.3f})')
                print(f'   ===> new regime{new_rgm.idx} is adopted')
                err = new_err
            else:
                self.rgm_c_idx = cand_regime.idx
        else:
            self.rgm_c_idx = cand_regime.idx
            new_Xc = Xc[:, -(self.lrprt + 1) :]
            for xt, yt in zip(new_Xc[:, :-1].T, new_Xc[:, 1:].T):
                cand_regime.learn_one(xt, yt)
        Vf = np.real(cand_regime.predict(Vc[:, -1], self.lstep + self.lrprt - 1))
        cond = np.abs(Vf).max(axis=1) > 1.8 * np.abs(Xc).max()
        Vf[cond] = Xc[cond, -(self.lstep + self.lrprt - 1) :]
        regime_storage.update(cand_regime)
        clock_time = time.monotonic() - _time

        # saving
        mdb.results['time'][mdb.tf : mdb.te] = clock_time
        mdb.results['rgm_idx'][mdb.tm : mdb.tc] = self.rgm_c_idx
        mdb.results['n_rgm'][mdb.tf : mdb.te] = len(regime_storage)
        mdb.logging_seq(Xc, Vc, Vf)

        return regime_storage, mdb
