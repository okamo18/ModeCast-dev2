from copy import deepcopy
from typing import Tuple

import numpy as np

from src.module.nlds import NLDS
from src.module.regime import Regime
from src.utils.metrics import rmse


class RegimeStorage:
    def __init__(self, regimes: list[Regime]) -> None:
        self.regimes = regimes
        self.assignment: dict
        self.n: int

    def __call__(self) -> list[Regime]:
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

    def init_regimes(self, regimes: list[Regime]) -> None:
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
    ) -> Tuple[Regime, np.ndarray, float]:
        n = Xc.shape[1]
        rgm_c = self[rgm_c_idx]
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

    def update(self, new_rgm: Regime) -> None:
        for i in range(len(self)):
            if self[i] == new_rgm:
                self.regimes[i] = new_rgm
                return

    def update_assignment(self, path: list[int], step: int, st: int, en: int) -> None:
        assignment = {path[0]: [[st, st + step]]}
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

    def __chk(self, regimes: list[Regime]) -> bool:
        for regime in regimes:
            if regime.intervals:
                return True
        return False

    def __get_next_regime(self, regimes: list[Regime], en: int) -> Regime:
        for regime in regimes:
            if regime.intervals and regime.intervals[0][0] == en:
                return regime
        raise ValueError("No next regime")
