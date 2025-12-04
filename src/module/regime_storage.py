from copy import deepcopy
from typing import Tuple, List, Dict, Iterator, Optional

import numpy as np

from src.module.nlds import NLDS
from src.module.regime import Regime
from src.utils.metrics import rmse


class RegimeStorage:
    def __init__(self, regimes: List[Regime]) -> None:
        # Regime のリスト
        self.regimes: List[Regime] = regimes
        # time index -> regime の割当情報
        self.assignment: Dict[int, List[List[int]]]
        # 全体の長さ
        self.n: int

    def __call__(self) -> List[Regime]:
        return self.regimes

    def __getitem__(self, idx: int) -> Regime:
        return self.regimes[idx]

    def __len__(self) -> int:
        return len(self.regimes)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RegimeStorage):
            for rgm, o_rgm in zip(self, other):
                if rgm != o_rgm:
                    return False
            return True
        return False

    def __iter__(self) -> "RegimeStorage":
        self._i = 0
        return self

    def __next__(self) -> Regime:
        if self._i == len(self):
            raise StopIteration
        regime = self.regimes[self._i]
        self._i += 1
        return regime

    def init_regimes(self, regimes: List[Regime]) -> None:
        self.regimes = regimes

    def get_path(self, return_nan: bool = False) -> np.ndarray:
        path = np.full(self.n, np.nan)
        for idx, intervals in self.assignment.items():
            for st, en in intervals:
                path[st:en] = idx
        return path if return_nan else path[~np.isnan(path)]


    def get_best_regime(
        self,
        Xc: np.ndarray,
        rgm_c_idx: int,
        err_th: float,
        Upsilon_c: Optional[np.ndarray] = None,  # ★ 追加：入力ウィンドウ（今は未使用）
    ) -> Tuple[Regime, np.ndarray, float]:
        """
        Xc        : (dxh, n)   現在の状態ウィンドウ列
        Upsilon_c : (duxh, n)  現在の入力ウィンドウ列（DMDc のときに使う予定）
        """
        n = Xc.shape[1]
        rgm_c = self[rgm_c_idx]

        # ★ 今の段階では Upsilon_c は使わず、既存どおり NLDS + DMD だけ
        nlds = NLDS(Xc)
        nlds.fit_X0(rgm_c)
        Vc = nlds.generate(rgm_c, n)
        err = rmse(Xc, Vc)
        if err < err_th:
            return rgm_c, Vc, err

        cand_regime = None
        min_err = np.inf
        for regime in self:
            nlds.fit_X0(regime)
            Vc = nlds.generate(regime, n)
            err = rmse(Xc, Vc)
            if err < min_err:
                cand_regime = regime
                min_err = err
        assert cand_regime is not None
        return deepcopy(cand_regime), Vc, min_err




    def _reconstruct_with_dmdc(
        self,
        regime: Regime,
        Xc: np.ndarray,
        Upsilon_c: np.ndarray,
    ) -> np.ndarray:
        """
        DMDc レジームを使って、観測区間 Xc を teacher forcing で再構成する。

        Xc       : (dxh, n)   ウィンドウ化された状態系列
        Upsilon_c: (duxh, n)  ウィンドウ化された入力系列
        戻り値   : (dxh, n)   再構成された系列 Vc
        """
        n = Xc.shape[1]
        dxh = Xc.shape[0]

        Vc = np.zeros_like(Xc)
        # 最初の時刻は真値をそのまま使う
        x = Xc[:, 0]
        Vc[:, 0] = x

        for k in range(n - 1):
            u = Upsilon_c[:, k]
            # Regime の中身は DMDc なので model.step が使える想定
            x = regime.model.step(x, u)
            Vc[:, k + 1] = x

        return Vc


    def update(self, new_rgm: Regime) -> None:
        for i in range(len(self)):
            if self[i] == new_rgm:
                self.regimes[i] = new_rgm
                return

    def update_assignment(self, path: List[int], step: int, st: int, en: int) -> None:
        assignment: Dict[int, List[List[int]]] = {path[0]: [[st, st + step]]}
        self.n = st + len(path)
        for i in range(1, len(path)):
            if path[i] != path[i - 1]:
                if path[i] not in assignment:
                    assignment[path[i]] = [[step * i + st, step * i + st + step]]
                else:
                    assignment[path[i]].append([step * i + st, step * i + st + step])
            else:
                assignment[path[i - 1]][-1][1] = min(step * i + st + step, en)
        self.assignment = {i: v for i, v in enumerate(assignment.values())}

    def append(self, regime: Regime) -> None:
        self.regimes.append(regime)

    def sync_intervals(self) -> None:
        for i in range(len(self)):
            if i in self.assignment:
                self.regimes[i].intervals = self.assignment[i]

    def reconstruct_by_only_one(self, X0: np.ndarray, st: int) -> np.ndarray:
        _regimes = deepcopy(self.regimes)
        c_itv = [0, st]
        c_x = X0
        pred = np.zeros((X0.shape[0], 0))
        while self.__chk(_regimes):
            rgm_c = self.__get_next_regime(_regimes, c_itv[1])
            c_itv = rgm_c.intervals.pop(0)
            # with_initial=True で [X0, x1, x2, ...] を返す
            pred = np.real(rgm_c.predict(c_x, c_itv[1] - c_itv[0], with_initial=True))
            pred = np.hstack([pred, pred])
            c_x = pred[:, -1]
        return pred

    def reconstruct_by_prev_one(self, X: np.ndarray, w: int, h: int) -> np.ndarray:
        pred = np.zeros((X.shape[0], 0))
        for i in range(w - 1, X.shape[1]):
            for regime in self.regimes:
                for st, en in regime.intervals:
                    if st <= i + h < en:
                        pred_i = regime.predict(X[:, i], 1)
                        pred = np.hstack([pred, pred_i])
        return pred

    def __chk(self, regimes: List[Regime]) -> bool:
        for regime in regimes:
            if regime.intervals:
                return True
        return False

    def __get_next_regime(self, regimes: List[Regime], en: int) -> Regime:
        for regime in regimes:
            if regime.intervals and regime.intervals[0][0] == en:
                return regime
        raise ValueError("No next regime")
