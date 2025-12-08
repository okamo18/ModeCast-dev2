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
from src.module.dmd import DMD
from src.module.dmdc import DMDc
from src.utils.preprocessor import create_variables, standardize ,create_variables_dmdc
from src.utils.visualizer import viz_snapshot

TEST_DMDC_ONLY = True  # DMDc 単体テストモードかどうか

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
    
    
    if TEST_DMDC_ONLY:
        # =========================================
        #  グローバル DMDc / DMD 素朴比較
        # =========================================
        d_state = state.shape[0]      # = 12
        h = cfg.model.h
        k0 = 50                       # 開始位置（適宜変えてOK）
        T = 50                        # 予測ステップ数
        
        # X_next の列数チェック（はみ出さないように）
        assert k0 + T < X_next.shape[1], "k0 + T がデータ長を超えています"

        # -------------------------------
        # 真の将来状態（12次元）の取り出し
        # -------------------------------
        true_future = np.zeros((d_state, T))
        for t in range(T):
            # X_next[:, k0 + t] は「ウィンドウ全体」（12*h 次元）
            x_mat_true = X_next[:, k0 + t].reshape(d_state, h)
            true_future[:, t] = x_mat_true[:, -1]  # 最後の列が「最新の状態」

        # ===============================
        # 1. グローバル DMDc
        # ===============================
        dmdc = DMDc(rank=20, trunc_th=cfg.model.trunc_th)
        dmdc.fit(X, X_next, Upsilon)

        # 初期ウィンドウ
        x_hist = X[:, k0].copy()          # (12*h,)
        u_hist = Upsilon[:, k0].copy()    # (6*h,)

        preds_dmdc = np.zeros((d_state, T))

        for t in range(T):
            # 1ステップ先のウィンドウ全体を予測
            x_hist = dmdc.step(x_hist, u_hist)      # (12*h,)

            # (12, h) に戻して「最新1ステップ」だけ抜く
            x_mat = x_hist.reshape(d_state, h)
            preds_dmdc[:, t] = x_mat[:, -1]

            # 入力ウィンドウは teacher forcing で真値から更新
            if k0 + t + 1 < Upsilon.shape[1]:
                u_hist = Upsilon[:, k0 + t + 1].copy()

        err_dmdc = rmse(true_future, preds_dmdc)
        print(f"[GLOBAL DMDc] multi-step RMSE (T={T}) = {err_dmdc:.4e}")
        print("  first pred (DMDc):", preds_dmdc[:, 0])
        print("  first true      :", true_future[:, 0])

        # ===============================
        # 2. グローバル DMD（制御なし）
        # ===============================
        dmd = DMD(trunc_th=cfg.model.trunc_th)
        dmd.fit(X, X_next)

        # DMD.predict は「ウィンドウ状態列」を返す想定
        # length=T+1, with_initial=True にして
        #   col 0: 初期 X[:, k0]
        #   col t: k0 から t ステップ進んだウィンドウ
        X_seq = dmd.predict(X[:, k0], length=T + 1, with_initial=True)  # (12*h, T+1)

        preds_dmd = np.zeros((d_state, T))
        for t in range(T):
            x_win = X_seq[:, t + 1]                   # 1ステップ先から順に
            x_mat = x_win.reshape(d_state, h)
            preds_dmd[:, t] = x_mat[:, -1]

        err_dmd = rmse(true_future, preds_dmd)
        print(f"[GLOBAL DMD ] multi-step RMSE (T={T}) = {err_dmd:.4e}")
        print("  first pred (DMD):", preds_dmd[:, 0])
        print("  first true      :", true_future[:, 0])

        # ここで終了：TSDMD / Regime には入らない
        return
    


    X_train = X[:, : n_est - h + 1]
    Y_train = X_next[:, : n_est - h + 1]
    Upsilon_train = Upsilon[:, : n_est - h + 1]

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
        use_control=False,    # ★ ここを True に
        rank=20,
    )

    # initializing
    regime_storage = tsdmd.initialize(X_train, Y_train, Upsilon_train)

    # forecasting
    s_st = n_est - lstep - lcurr + 1
    s_en = n - lstep - lcurr - lrprt + 2
    assert s_st >= 0

    mdb = MDB(cfg, X)
    for tm in range(s_st, s_en, lrprt):
        tc = tm + lcurr
        tf = tm + lcurr + lstep - 1
        te = tm + lcurr + lstep + lrprt - 1

        # mdb にはこれまで通り「生時系列側のインデックス」で渡す
        mdb.set_params(tm=tm + h - 1, tc=tc, tf=tf, te=te)

        # -----------------------------
        # 1. 状態ウィンドウ Xc （今まで通り）
        # -----------------------------
        Xc = X[:, tm : tc - h + 1]   # shape: (d*h, lcurr-h+1)

        # -----------------------------
        # 2. 制御ウィンドウ Upsilon_c（現在区間）
        #    ※ use_control=False のときは None のままで OK
        # -----------------------------
        if tsdmd.use_control:
            Upsilon_c = Upsilon[:, tm : tc - h + 1]  # Xc と同じ列を取る
        else:
            Upsilon_c = None

        # -----------------------------
        # 3. 将来区間の制御 Upsilon_future を用意
        #    length = lstep + lrprt - 1 ステップぶんの遷移で使う入力列
        #    Regime.predict(with_initial=True) は
        #      - 出力列数 : length
        #      - 使う u の本数 : length-1
        # -----------------------------
        length = lstep + lrprt - 1

        if tsdmd.use_control:
            # X の列インデックスで「次のウィンドウ」の位置からスタート
            #   Xc の最後の列 index = (tc - h)
            #   その次の列 = (tc - h + 1)
            start_u = tc - h + 1
            end_u = start_u + (length - 1)

            # 念のため安全のためにクリップしておく（はみ出たぶんは Regime 側で u=0 フォールバック）
            max_cols = Upsilon.shape[1]
            if end_u > max_cols:
                end_u = max_cols

            if start_u < end_u:
                Upsilon_future = Upsilon[:, start_u:end_u]
            else:
                # もし範囲がおかしければゼロ入力にフォールバック
                Upsilon_future = None
        else:
            Upsilon_future = None

        # -----------------------------
        # 4. TSDMD でレジーム選択＋将来予測
        # -----------------------------
        regime_storage, mdb = tsdmd.forecast(
            Xc,
            regime_storage,
            mdb,
            Upsilon_c=Upsilon_c,
            Upsilon_future=Upsilon_future,
        )

        # -----------------------------
        # 5. 誤差計算はこれまで通り
        # -----------------------------
        err_c = rmse(mdb.Xc, mdb.Vc)
        err_f = rmse(data_state[:, tf:te], mdb.Vf[:, lstep - 1 :])

        # saving
        mdb.results["err_c"][tm:tc] = err_c
        mdb.results["err_f"][tf:te] = err_f

        show_snapshot_info(logger, mdb)

        fig = viz_snapshot(
            data_state,
            np.hstack([mdb.Vc, mdb.Vf]).T,
            h,
            tm,
            tc,
            tf,
            te,
            err_c,
            show=cfg.viz,
        )
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
