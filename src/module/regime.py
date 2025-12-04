import numpy as np

from src.module.dmd import DMD
#from src.module.dmdc import DMDc


class Regime(DMD):
    def __init__(self, trunc_th: float, idx: int = 0) -> None:
        super().__init__(trunc_th=trunc_th)
        self.intervals: list = []
        self.idx = idx

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

    def get_modes(self, X0: np.ndarray, length: int) -> np.ndarray:
        Lamb, W = np.linalg.eig(self.A_tilde)
        Phi = self.U @ W
        modes = np.zeros([Lamb.shape[0], length], dtype="complex")
        b = np.linalg.pinv(Phi) @ X0
        for i in range(length):
            modes[:, i] = np.diag(Lamb**i) @ b
        return modes
