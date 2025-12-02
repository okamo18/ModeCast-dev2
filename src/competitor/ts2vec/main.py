import pandas as pd
import numpy as np
import argparse
import os
import shutil
from sklearn.linear_model import LinearRegression

import sys

sys.path.append("_src")
import utils
import time
import dill

from ts2vec import TS2Vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input/Output Data
    parser.add_argument("--input_tag", type=str)
    parser.add_argument("--out_dir", type=str, default="out/tmp/")
    parser.add_argument("--import_type", type=str, default="clean_data_rul")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--feature_cols", type=str)
    parser.add_argument("--label_col", type=str)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--use_dict_inputs", action="store_true")
    parser.add_argument("--use_sequence_inputs", action="store_true")
    parser.add_argument("--split_ind", type=int, default=0)  # from 0 to 5 or 10

    # Experimental setting
    parser.add_argument("--model_name", type=str, default="linear")
    parser.add_argument("--emb_n_dim", type=int, default=320)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)


    args = parser.parse_args()
    # make output dir
    outputdir = args.out_dir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # Original data
    train_dataset, test_dataset = utils.import_dataset(args)

    # Z-normalization
    train_dataset, test_dataset = utils.znorm_train_test_datasets(train_dataset, test_dataset, args)

    # Transform dataset for input
    print("Train samples")
    train_data, train_result_df = utils.make_samples(
        train_dataset,
        args,
    )  # data: dict object

    print("\nTest samples")
    test_data, test_result_df = utils.make_samples(
        test_dataset,
        args,
    )  # data: dict object

    train_x = train_data["features"] # (seq x n_window x window size x ndim)
    train_x_per_seq_flat = [np.concatenate([tx[0,:,:],tx[1:,-1,:]]) for i, tx in enumerate(train_x)]
    # train_x_all_flat = np.concatenate(train_x_per_seq_flat,axis=0)
    train_y = np.concatenate(train_data["label"],axis=0) # (seq x n_window x 1)

    test_x = test_data["features"]
    test_x_per_seq_flat = [np.concatenate([tx[0,:,:],tx[1:,-1,:]]) for tx in test_x]
    # test_x_all_flat = np.concatenate(test_x_perseq_flat,axis=0)
    test_y = np.concatenate(test_data["label"],axis=0)

    max_len = np.max([len(tx) for tx in train_x_per_seq_flat] + [len(tx) for tx in test_x_per_seq_flat])
    input_dims = train_x_per_seq_flat[0].shape[1]
    
    train_x_per_seq_fillna=[]
    for tx in train_x_per_seq_flat:
        temp = np.full((max_len,input_dims) ,np.nan)
        temp[-tx.shape[0]:] = tx
        train_x_per_seq_fillna.append(temp) 
    train_x_per_seq_fillna = np.array(train_x_per_seq_fillna)

    test_x_per_seq_fillna=[]
    for tx in test_x_per_seq_flat:
        temp = np.full((max_len,input_dims) ,np.nan)
        temp[-tx.shape[0]:] = tx
        test_x_per_seq_fillna.append(temp) 
    test_x_per_seq_fillna = np.array(test_x_per_seq_fillna)
    
    params = vars(args)

    # build ts2vec
    tic = time.process_time()

    # # Load the ECG200 dataset from UCR archive
    # train_data, train_labels, test_data, test_labels = datautils.load_UCR('ECG200')
    # # (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)

    # Train a TS2Vec model
    model = TS2Vec(
        input_dims=input_dims,
        device='cuda',
        output_dims=params["emb_n_dim"],
        lr=params["lr"],
        batch_size=params["batch_size"],
    )

    loss_log = model.fit(
        train_x_per_seq_fillna,
        verbose=True,
        n_epochs=params["epochs"],
    )

    # Compute timestamp-level representations for test set
    train_window_repr = model.encode(
        train_x_per_seq_fillna,
        causal=True,
        sliding_length=1,
        sliding_padding=params["window_size"]
    )  # n_instances x n_timestamps x output_dims

    train_window_repr_wo_nan_flat = np.concatenate([t_repr[-tx.shape[0]:] for t_repr, tx in zip(train_window_repr,train_x)],axis=0)

    # train_flat_x = train_window_repr_wo_nan_flat.reshape(,-1)

    if params["model_name"] == "linear":
        reg = LinearRegression().fit(train_window_repr_wo_nan_flat, train_y)
    else:
        NotImplementedError

    learning_time = time.process_time() - tic
    params["learning_time"] = learning_time
    utils.save_as_json(params, params["out_dir"] + "/setting.json")

    # predict
    # time-to-event prediction
    tic = time.process_time()
    pred_train = reg.predict(train_window_repr_wo_nan_flat)
    pred_time_train = time.process_time() - tic

    tic = time.process_time()
    # Compute timestamp-level representations for test set
    test_window_repr = model.encode(
        test_x_per_seq_fillna,
        causal=True,
        sliding_length=1,
        sliding_padding=params["window_size"]
        )  # n_instances x n_timestamps x output_dims

    test_window_repr_wo_nan = np.concatenate([t_repr[-tx.shape[0]:] for t_repr, tx in zip(test_window_repr,test_x)],axis=0)
    pred_test = reg.predict(test_window_repr_wo_nan)
    
    # Compute instance-level representations for test set
    train_seq_repr = model.encode(
        train_x_per_seq_fillna,
        encoding_window='full_series')  # n_instances x output_dims
    test_seq_repr = model.encode(
        test_x_per_seq_fillna, 
        encoding_window='full_series')  # n_instances x output_dims
    pred_time_test = time.process_time() - tic

    
    # write result
    train_result_df["pred"] = pred_train
    train_result_df["test"] = 0
    test_result_df["pred"] = pred_test
    test_result_df["test"] = 1
    result_df = pd.concat([train_result_df, test_result_df], axis=0)
    result_df["method"] = f"ts2vec_{params['model_name']}"

    mae = utils.mean_absolute_error(test_y, pred_test)
    rmse = utils.mean_squared_error(test_y, pred_test, squared=False)

    summary_dict = {
        "pred_time_train": pred_time_train,
        "pred_time_test": pred_time_test,
        "MAE": mae,
        "RMSE": rmse,
    }
    emb_results={
        "emb_train":train_window_repr,
        "emb_test":test_window_repr,
        "emb_train_seq":train_seq_repr,
        "emb_test_seq":test_seq_repr,
    }

    # save result
    result_df.to_csv(params["out_dir"] + "/result.csv.gz", index=False)
    utils.save_as_json(summary_dict, params["out_dir"] + "/summary.json")
    dill.dump(emb_results, open(params["out_dir"] + "/emb_results.dill", "wb"))


    print("Summary:")
    print(f"MAE:{mae}")
    print(f"RMSE:{rmse}")
    print(params["out_dir"])


