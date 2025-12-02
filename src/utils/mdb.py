import numpy as np
from omegaconf import DictConfig


class MDB:
    def __init__(self, cfg: DictConfig, X: np.ndarray) -> None:
        self.n = X.shape[1] + cfg.model.h
        self.d = X.shape[0] // cfg.model.h
        self.h = cfg.model.h
        self.dh = X.shape[0]
        self.lstep = cfg.model.lstep
        self.metadata = {
            "lcurr": cfg.model.lcurr,
            "lstep": cfg.model.lstep,
            "lrprt": cfg.model.lrprt,
            "err_th": cfg.model.err_th,
            "trunc_th": cfg.model.trunc_th,
            "h": cfg.model.h,
            "dh": X.shape[0],
        }
        self.results = {
            "time": np.nan * np.zeros(self.n),
            "n_rgm": np.nan * np.zeros(self.n),
            "rgm_idx": np.nan * np.zeros(self.n),
            "err_c": np.nan * np.zeros(self.n),
            "err_f": np.nan * np.zeros(self.n),
        }
        for d_i in range(self.dh):
            self.results[f"X{d_i}"] = np.nan * np.zeros(self.n)
            self.results[f"Ve{d_i}"] = np.nan * np.zeros(self.n)
            self.results[f"Vs{d_i}"] = np.nan * np.zeros(self.n)
        self.tm: int
        self.tc: int
        self.tf: int
        self.te: int

    def set_params(self, **parameters: dict) -> "MDB":
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @property
    def time(self) -> np.ndarray:
        return self.results["time"][~np.isnan(self.results["time"])]

    @property
    def rgm_idx(self) -> np.ndarray:
        return self.results["rgm_idx"][~np.isnan(self.results["rgm_idx"])].astype(np.int8)

    @property
    def err_c(self) -> np.ndarray:
        return self.results["err_c"][~np.isnan(self.results["err_c"])]

    @property
    def err_f(self) -> np.ndarray:
        return self.results["err_f"][~np.isnan(self.results["err_f"])]

    @property
    def X(self) -> np.ndarray:
        return self.__get_X()

    @property
    def Ve(self) -> np.ndarray:
        return self.__get_Ve()

    @property
    def Vs(self) -> np.ndarray:
        return self.__get_Vs()

    @property
    def full_X(self) -> np.ndarray:
        return self.__get_X(full=True)

    @property
    def full_Ve(self) -> np.ndarray:
        return self.__get_Ve(full=True)

    @property
    def full_Vs(self) -> np.ndarray:
        return self.__get_Vs(full=True)

    def logging_seq(self, Xc: np.ndarray, Vc: np.ndarray, Vf: np.ndarray) -> None:
        self.Xc = Xc[[i for i in range(self.h - 1, self.dh, self.h)], :]
        self.Vc = Vc[[i for i in range(self.h - 1, self.dh, self.h)], :]
        self.Vf = Vf[[i for i in range(self.h - 1, self.dh, self.h)], :]
        self.__logging_Xc(Xc)
        self.__logging_Vc(Vc)
        self.__logging_Vf(Vf)

    def __get_X(self, full: bool = False) -> np.ndarray:
        if full:
            return np.array([self.results[f"X{i}"] for i in range(self.dh)])
        else:
            return np.array([self.results[f"X{i}"] for i in range(self.h - 1, self.dh, self.h)])

    def __get_Ve(self, full: bool = False) -> np.ndarray:
        if full:
            return np.array([self.results[f"Ve{i}"] for i in range(self.dh)])
        else:
            return np.array([self.results[f"Ve{i}"] for i in range(self.h - 1, self.dh, self.h)])

    def __get_Vs(self, full: bool = False) -> np.ndarray:
        if full:
            return np.array([self.results[f"Vs{i}"] for i in range(self.dh)])
        else:
            return np.array([self.results[f"Vs{i}"] for i in range(self.h - 1, self.dh, self.h)])

    def __logging_Xc(self, Xc: np.ndarray) -> None:
        for i in range(self.dh):
            self.results[f"X{i}"][self.tm : self.tc] = Xc[i, :]

    def __logging_Vc(self, Vc: np.ndarray) -> None:
        for i in range(self.dh):
            self.results[f"Ve{i}"][self.tm : self.tc] = Vc[i, :]

    def __logging_Vf(self, Vf: np.ndarray) -> None:
        for i in range(self.dh):
            self.results[f"Vs{i}"][self.tf : self.te] = Vf[i, self.lstep - 1 :]
