import datetime
import logging
from typing import Optional

import numpy as np

from src.utils.mdb import MDB
from src.utils.metrics import eval_metrics_reg, eval_metrics_seg


def show_input(logger: logging.Logger, data: np.ndarray) -> None:
    logger.info(f"[{datetime.datetime.now()}]")
    logger.info("-------------------------------")
    logger.info("          Input Data           ")
    logger.info("-------------------------------")
    logger.info(f"   dimension:\t{data.shape[0]}")
    logger.info(f"   length:\t{data.shape[1]}")
    logger.info("-------------------------------\n")


def show_metrics(
    logger: logging.Logger,
    true_path: Optional[list[int]] = None,
    true_curve: Optional[np.ndarray] = None,
    pred_path: Optional[np.ndarray] = None,
    pred_curve: Optional[np.ndarray] = None,
) -> None:
    if true_path is not None and pred_path is not None:
        logger.info("-------------------------------")
        logger.info("             Score             ")
        logger.info("-------------------------------")
        for name, score in eval_metrics_seg(true_path, pred_path).items():
            logger.info(f"   {name}: {score:.3f}")
        logger.info("-------------------------------\n")
    if true_curve is not None and pred_curve is not None:
        logger.info("-------------------------------")
        logger.info("             Error             ")
        logger.info("-------------------------------")
        for name, err in eval_metrics_reg(true_curve, pred_curve).items():
            logger.info(f"   {name}: {err:.3f}")
        logger.info("-------------------------------\n")


def show_snapshot_info(
    logger: logging.Logger,
    mdb: MDB,
    show: bool = True,
) -> None:
    h = mdb.metadata["h"]
    if show:
        logger.info("---------------------------------------------")
        logger.info(f"  given ...... Xc = [x({mdb.tm - h + 1}), ..., x({mdb.tc - 1})]")
        logger.info(f"  forecast ... Xf = [x({mdb.tf}), ..., x({mdb.te - 1})]")
        logger.info("---------------------------------------------")
        logger.info(f"   #regime {mdb.rgm_idx[-1] + 1}")
        logger.info(f"   time: {mdb.time[-1]:.5f} [sec]")
        logger.info(f"   rmse: |Xc-Vc|={mdb.err_c[-1]:.3f}, |Xf-Vf|={mdb.err_f[-1]:.3f}")
        logger.info("---------------------------------------------\n")
