import logging

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.module.tsdmd import TSDMD
from src.utils import change2darkmode
from src.utils.displayer import show_input, show_metrics, show_snapshot_info
from src.utils.io_helper import IOHelper
from src.utils.mdb import MDB
from src.utils.metrics import rmse
from src.module.dmdc import DMDc
from src.utils.preprocessor import create_variables, standardize ,create_variables_dmdc
from src.utils.visualizer import viz_snapshot


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # ------------------------------------ #
    #            preprocessing             #
    # ------------------------------------ #
    ioh = IOHelper(cfg.io)
    ioh.init_dir()
    
    #data = standardize(ioh.import_arr_data(fn=cfg.io.fn, window=15)).T #Modecast用
    
    df = pd.read_csv( "data/Robot/robot_csv/recording_2021_12_15_20H_29M.csv", dtype=np.float64)
    data_raw = df.to_numpy(dtype=np.float64)
    data = standardize(data_raw).T
    data = data[:, ::100]  # 250Hz → 25Hz に間引く
    data = np.ascontiguousarray(data)
    q_indices = [1 + 3 * i for i in range(6)]   # 1,4,7,10,13,16
    qd_indices = [2 + 3 * i for i in range(6)]  # 2,5,8,11,14,17
    tau_indices = [3 + 3 * i for i in range(6)] # 3,6,9,12,15,18

    state = np.vstack([
        data[q_indices, :],
        data[qd_indices, :],
    ])  # shape: (12, n)

    control = data[tau_indices, :]  # shape: (6, n)
    state = np.ascontiguousarray(state)
    data_state = state
    control = np.ascontiguousarray(control)


    print("state shape:", state.shape)    # (12, n)
    print("control shape:", control.shape)  # (6, n)

    logger = logging.getLogger(__name__)

    h = cfg.model.h
    w = cfg.model.window
    n_est = cfg.model.train_size
    lstep = cfg.model.lstep
    lrprt = cfg.model.lrprt
    lcurr = max(cfg.model.lcurr, lstep + lrprt + h - 2)
    d, n = data.shape

    #X, Y = create_variables(data=data, h=h) #Modecast用

    X, X_next, Upsilon = create_variables_dmdc(state, control, h=h)
    

    # =========================================
    #  DMDc 単体テスト（A, B がちゃんと出るか確認）
    # =========================================
    dmdc = DMDc(rank=50)  # ランクは適当。あとで調整してOK
    dmdc.fit(X, X_next, Upsilon)

    # 最初の1ステップでどれくらい当たってるかを見る
    x0 = X[:, 0]
    u0 = Upsilon[:, 0]
    x1_true = X_next[:, 0]
    x1_pred = dmdc.step(x0, u0)

    err_one_step = np.linalg.norm(x1_true - x1_pred)
    print(f"[DMDc test] one-step prediction error (norm) = {err_one_step:.4e}")


    # =========================================
    #  DMDc multi-step test
    # =========================================
    T = 50    # 何ステップ先まで見るか（お好みで）
    t0 = 100  # 予測をスタートする時刻（h 以上、n-T-1 以下にしておく）

    # ① 初期の「状態ウィンドウ」と「入力ウィンドウ」を作る
    # state: (12, n), control: (6, n)
    # 遅延長 h ステップぶんを縦に並べたベクトルにする
    x_hist = state[:, t0 - h + 1 : t0 + 1].flatten()      # shape: (12*h,)
    u_hist = control[:, t0 - h + 1 : t0 + 1].flatten()    # shape: (6*h,)

    preds = []  # 各ステップの「最新状態」だけを貯める

    for t in range(T):
        # ② 1ステップ先の「ウィンドウ全体」を予測
        x_hist_next = dmdc.step(x_hist, u_hist)  # shape: (12*h,)

        # ③ そのウィンドウの一番新しい状態（最後の12要素）を抜き出す
        x_new = x_hist_next[-12:]
        preds.append(x_new)

        # ④ 次のステップ用にウィンドウを更新
        #    「古い1ステップ分(先頭12)を捨てて、末尾に x_new をくっつける」
        x_hist = np.hstack([x_hist[12:], x_new])

        # 入力も同様に「1ステップずらしたウィンドウ」に更新
        #   → ここでは真の control から取ってくる
        u_window = control[:, t0 - h + 2 + t : t0 + 2 + t]  # 次の h ステップ分
        u_hist = u_window.flatten()

    preds = np.stack(preds, axis=1)  # shape: (12, T)

    # ⑤ 真の状態（12次元）との RMSE を計算
    true_future = state[:, t0 + 1 : t0 + 1 + T]  # shape: (12, T)

    from src.utils.metrics import rmse
    err_multi = rmse(true_future, preds)
    print(f"[DMDc test] multi-step RMSE (T={T}) = {err_multi:.4e}")

    # ここで一旦終了して、後続の TSDMD/ModeCast はスキップ
    return 

    


    X_train = X[:, : n_est - h + 1]
    Y_train = X_next[:, : n_est - h + 1]

    show_input(logger, data)
    if cfg.dark_mode:
        change2darkmode()

    tsdmd = TSDMD(
        n=n_est,
        d=d,
        h=h,
        k=cfg.model.init_n_rgm,
        w=w,
        lcurr=cfg.model.lcurr,
        lstep=cfg.model.lstep,
        lrprt=cfg.model.lrprt,
        err_th=cfg.model.err_th,
        trunc_th=cfg.model.trunc_th,
        max_iter=cfg.model.max_iter,
    )

    # initializing
    regime_storage = tsdmd.initialize(X_train, Y_train)

    # forecasting
    s_st = n_est - lstep - lcurr + 1
    s_en = n - lstep - lcurr - lrprt + 2
    assert s_st >= 0

    mdb = MDB(cfg, X)
    for tm in range(s_st, s_en, lrprt):
        tc = tm + lcurr
        tf = tm + lcurr + lstep - 1
        te = tm + lcurr + lstep + lrprt - 1
        mdb.set_params(tm=tm + h - 1, tc=tc, tf=tf, te=te)
        Xc = X[:, tm : tc - h + 1]
        regime_storage, mdb = tsdmd.forecast(Xc, regime_storage, mdb)
        err_c = rmse(mdb.Xc, mdb.Vc)
        err_f = rmse(data_state[:, tf:te], mdb.Vf[:, lstep - 1 :])

        # saving
        mdb.results["err_c"][tm:tc] = err_c
        mdb.results["err_f"][tf:te] = err_f

        show_snapshot_info(logger, mdb)

        fig = viz_snapshot(data_state, np.hstack([mdb.Vc, mdb.Vf]).T, h, tm, tc, tf, te, err_c, show=cfg.viz)
        ioh.savefig(fig, f"snapshot/[{tm} : {tf+lrprt}] fit.png")

    show_metrics(
        logger=logger,
        true_curve=data_state[:, n_est : n - (n - n_est) % lrprt],
        pred_curve=mdb.Vs[:, ~np.isnan(mdb.Vs).any(axis=0)],
    )

    ioh.savepkl(regime_storage, "regime_storage.pkl")
    ioh.savepkl(mdb, "MDB.pkl")


if __name__ == "__main__":
    main()
