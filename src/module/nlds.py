import lmfit
import numpy as np

from src.module.regime import Regime

XTLi = 0.1
FTLi = 0.1
MAXFEVi = 20


class NLDS:
    def __init__(self, X: np.ndarray) -> None:
        self.X = X
        self.X0 = X[:, 0].copy()
        self.dh = X.shape[0]

    def fit_X0(self, model: "Regime", dps: int = 1) -> "NLDS":
        return nl_fit(self, model, dps)

    def generate(self, model: "Regime", n: int) -> np.ndarray:
        return model.predict(self.X0, n, with_initial=True)


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
