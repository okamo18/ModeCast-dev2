import os
import pickle
import shutil
from importlib import import_module
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig


class IOHelper:
    def __init__(self, io_cfg: DictConfig) -> None:
        self.input_dir = io_cfg.input_dir
        self.out_dir = io_cfg.out_dir

    def init_dir(self) -> None:
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)
        if os.path.isdir(self.out_dir + "/snapshot"):
            shutil.rmtree(self.out_dir + "/snapshot")
        os.mkdir(self.out_dir + "/snapshot")

    def import_arr_data(self, fn: str, window: int = 1, tag: Optional[str] = None) -> pd.DataFrame:
        dataset_module = import_module(f"data.{self.input_dir}")
        if tag is None:
            arr = dataset_module.load_arr_data(fn, window=window)
        else:
            raise NotImplementedError

        return arr

    def savefig(self, fig: Figure, name: str) -> None:
        fig.savefig(self.out_dir + name)
        plt.close()

    def savepkl(self, obj: Any, name: str) -> None:
        if "." not in name:
            name += ".pkl"
        f = open(self.out_dir + name, "wb")
        pickle.dump(obj, f)
        f.close()

    def loadpkl(self, name: str) -> Any:
        if "." not in name:
            name += ".pkl"
        try:
            f = open(self.out_dir + name, "rb")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.out_dir + name}")
        obj = pickle.load(f)
        f.close()

        return obj
