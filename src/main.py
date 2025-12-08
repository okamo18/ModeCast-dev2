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
    #  グローバル DMD / DMDc 比較（案A）
    # =========================================
    DO_GLOBAL_COMPARE = False  # ← 終わったら False にしてもいい

    if DO_GLOBAL_COMPARE:


        d_state = state.shape[0]   # 12
        h = cfg.model.h           # 遅延長は config と合わせる
        T = 50
        k0 = 50                   # 予測スタート位置（とりあえず固定）

        # 共通の遅延埋め込み（状態+入力）
        X, X_next, Upsilon = create_variables_dmdc(state, control, h=h)

        # 真の将来状態（12次元）だけを取り出す
        true_future = np.zeros((d_state, T))
        for t in range(T):
            true_mat = X_next[:, k0 + t].reshape(d_state, h)
            true_future[:, t] = true_mat[:, -1]

        # ---------------- DMDc: rank を変えながら ----------------
        rank_list = [5, 10, 15, 20, 50]

        for rank in rank_list:
            dmdc = DMDc(rank=rank, trunc_th=0.99)
            dmdc.fit(X, X_next, Upsilon)

            # 初期ウィンドウ（12*h 次元）
            x_hist = X[:, k0].copy()
            u_hist = Upsilon[:, k0].copy()

            preds_c = np.zeros((d_state, T))

            for t in range(T):
                # 1ステップ先のウィンドウ全体を予測
                x_hist = dmdc.step(x_hist, u_hist)       # (12*h,)

                # (12, h) に reshape して最後の列＝最新状態を取り出す
                x_mat = x_hist.reshape(d_state, h)
                x_new = x_mat[:, -1]
                preds_c[:, t] = x_new

                # teacher forcing で入力も1ステップ先に進める
                if k0 + t + 1 < Upsilon.shape[1]:
                    u_hist = Upsilon[:, k0 + t + 1].copy()

            err_c = rmse(true_future, preds_c)
            print(
                f"[GLOBAL DMDc] h={h}, rank={rank}, "
                f"multi-step RMSE (T={T}) = {err_c:.4e}"
            )
            print("  first pred (DMDc):", preds_c[:, 0])
            print("  first true      :", true_future[:, 0])

        # ---------------- DMD: 1 回だけ ----------------
        dmd = DMD(trunc_th=0.99)
        dmd.fit(X, X_next)

        # DMD の予測は X のウィンドウ全体（12*h 次元）で返ってくる前提
        X_seq = dmd.predict(X[:, k0], n=T + 1, with_initial=True)  # (12*h, T+1)

        preds_d = np.zeros((d_state, T))
        for t in range(T):
            x_mat = X_seq[:, t + 1].reshape(d_state, h)  # t+1 が 1ステップ先
            preds_d[:, t] = x_mat[:, -1]

        err_d = rmse(true_future, preds_d)
        print(
            f"[GLOBAL DMD ] h={h}, multi-step RMSE (T={T}) = {err_d:.4e}"
        )
        print("  first pred (DMD):", preds_d[:, 0])
        print("  first true      :", true_future[:, 0])

        # ここで一旦終了（TSDMD / ModeCast 本体は動かさない）
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
        rank=10,
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
