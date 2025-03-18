#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

def build_features_for_one_model(model_dir: str, labels_required=True):
    challenge_with_id_csv = os.path.join(model_dir, "challenge_with_id.csv")
    df_challenge = pd.read_csv(challenge_with_id_csv)
    print(f"[{model_dir}] challenge_with_id.csv shape: {df_challenge.shape}")

    if labels_required:
        challenge_label_csv = os.path.join(model_dir, "challenge_label.csv")
        df_label = pd.read_csv(challenge_label_csv)
        print(f"[{model_dir}] challenge_label.csv shape: {df_label.shape}")
        n = min(len(df_challenge), len(df_label))
        if len(df_challenge) != len(df_label):
            print(f"[{model_dir}] Warning: Row count mismatch; using first {n} rows.")
            df_challenge = df_challenge.iloc[:n]
            df_label = df_label.iloc[:n]
        y = df_label["is_train"].values
    else:
        y = None

    trans_csv = os.path.join(model_dir, "trans.csv")
    account_csv = os.path.join(model_dir, "account.csv")
    if os.path.exists(trans_csv) and os.path.exists(account_csv):
        df_trans = pd.read_csv(trans_csv)
        df_account = pd.read_csv(account_csv)
        disp_csv = os.path.join(model_dir, "disp.csv")
        df_disp = pd.read_csv(disp_csv) if os.path.exists(disp_csv) else None

        df_merge1 = pd.merge(df_challenge, df_trans, on="trans_id", how="left")
        print(f"[{model_dir}] After merge with trans.csv: {df_merge1.shape}")

        merge_key = "account_id" if "account_id" in df_merge1.columns else "account_id_x"
        df_merge2 = pd.merge(df_merge1, df_account, left_on=merge_key, right_on="account_id", how="left")
        print(f"[{model_dir}] After merge with account.csv: {df_merge2.shape}")

        if df_disp is not None:
            merge_key_disp = "account_id" if "account_id" in df_merge2.columns else "account_id_x"
            df_disp_agg = (
                df_disp.groupby("account_id")
                .agg(
                    disp_count=("disp_id", "count"),
                    client_count=("client_id", "nunique"),
                )
                .reset_index()
            )
            df_merge2 = pd.merge(df_merge2, df_disp_agg, left_on=merge_key_disp, right_on="account_id", how="left")
            print(f"[{model_dir}] After merge with aggregated disp.csv: {df_merge2.shape}")

        if "amount" in df_merge2.columns:
            df_merge2["log_amount"] = np.log1p(df_merge2["amount"].fillna(0).clip(lower=0))

        df_features = df_merge2.copy()
    else:
        print(f"[{model_dir}] Warning: Missing base files. Using only challenge_with_id.csv for features.")
        df_features = df_challenge.copy()

    print(f"[{model_dir}] Final feature rows: {df_features.shape[0]}")
    return df_features, y

def build_dataset_for_all_train_models(base_dir: str):
    import glob
    import os
    
    model_dirs = glob.glob(os.path.join(base_dir, "clavaddpm_*"))
    X_list, y_list = [], []
    for md in model_dirs:
        if not os.path.isdir(md):
            continue
        try:
            X_cur, y_cur = build_features_for_one_model(md)
            print(f"[{md}] Returned feature rows: {X_cur.shape[0]}, label rows: {len(y_cur)}")
            X_list.append(X_cur)
            y_list.append(y_cur)
        except Exception as e:
            print(f"Error building features for {md}: {e}")
            continue
    
    if X_list:
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        return X_all, y_all
    else:
        print("No valid data found.")
        return None, None


def main():
    train_dir = "clavaddpm_white_box/train"
    
    X_all, y_all = build_dataset_for_all_train_models(train_dir)
    if X_all is not None:
        print("Final dataset shape:", X_all.shape, "Labels shape:", y_all.shape)
    else:
        print("Could not build dataset from train models.")

if __name__ == "__main__":
    main()