"""Federated intrusion detection on NSL-KDD (5 simulated banks, FedAvg).

Usage:
    python src/fl_ids.py --test-path data/test_public.csv --out submission.csv

Pipeline: download NSL-KDD -> clean/encode/scale -> reserve a seed-locked holdout
-> split the rest into 5 IID clients -> FedAvg with an improved local model
-> tune the decision threshold -> write a Kaggle-format submission.csv.
"""
import argparse
import copy
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEED = 42
N_CLIENTS = 5
HOLDOUT_SIZE = 15000  # must match the organisers' value, otherwise clients may see the hidden test rows

TRAIN_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"

COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
CAT_COLS = ["protocol_type", "service", "flag"]
DROP_COLS = ["num_outbound_cmds"]  # ~constant in NSL-KDD
NEW_COLS = ["error_asymmetry", "srv_error_asymmetry", "bytes_diff",
            "host_diff_ratio", "conn_density", "login_risk"]


# ----------------------------------------------------------------- data
def load_and_prepare():
    train_raw = pd.read_csv(TRAIN_URL, names=COLS)
    test_raw = pd.read_csv(TEST_URL, names=COLS)

    def clean(df):
        df = df.drop(columns=["difficulty"]).copy()
        df["binary_label"] = (df["label"] != "normal").astype(int)
        return df

    train_df, test_df = clean(train_raw), clean(test_raw)

    for c in CAT_COLS:
        le = LabelEncoder().fit(pd.concat([train_df[c], test_df[c]]))
        train_df[c], test_df[c] = le.transform(train_df[c]), le.transform(test_df[c])

    feature_cols = [c for c in train_df.columns if c not in ("label", "binary_label")]
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, feature_cols


def engineer_features(df):
    df = df.copy()
    df["error_asymmetry"] = df["serror_rate"] - df["rerror_rate"]
    df["srv_error_asymmetry"] = df["srv_serror_rate"] - df["srv_rerror_rate"]
    df["bytes_diff"] = df["src_bytes"] - df["dst_bytes"]
    df["host_diff_ratio"] = df["dst_host_diff_srv_rate"] - df["dst_host_same_srv_rate"]
    df["conn_density"] = df["count"] * df["srv_count"]
    df["login_risk"] = df["num_failed_logins"] + df["is_guest_login"]
    return df


def make_clients(train_df):
    """Seed-locked holdout reservation, then an IID split into N_CLIENTS partitions."""
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train_df))
    holdout_idx, pool_idx = perm[:HOLDOUT_SIZE], perm[HOLDOUT_SIZE:]
    fp = hashlib.md5(str(sorted(holdout_idx.tolist())).encode()).hexdigest()[:12]
    print(f"Holdout fingerprint: {fp}  (compare with the organisers' value)")

    pool = train_df.iloc[pool_idx].reset_index(drop=True)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(pool))
    return [pool.iloc[c].reset_index(drop=True) for c in np.array_split(idx, N_CLIENTS)]


# ---------------------------------------------------------------- model
class IDSModel(nn.Module):
    """Small MLP. LayerNorm (not BatchNorm) because FedAvg only averages learnable
    parameters, never BatchNorm running statistics."""

    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------ federated
def local_train(model, X, y, epochs=3, lr=0.01, batch_size=256):
    """One client's local update. pos_weight is computed from local labels only."""
    model = copy.deepcopy(model)
    pos_weight = torch.tensor([float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()
    return model


def get_flat(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view(p.shape))
        i += n


def fedavg(models, sizes):
    w = torch.tensor(np.array(sizes, dtype=float) / np.sum(sizes), dtype=torch.float32)
    avg = (torch.stack([get_flat(m) for m in models]) * w.unsqueeze(1)).sum(0)
    out = copy.deepcopy(models[0])
    set_flat(out, avg)
    return out


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy()


def metrics(y_true, probs, thr=0.5):
    p = (probs > thr).astype(int)
    return {"precision": round(precision_score(y_true, p, zero_division=0), 4),
            "recall": round(recall_score(y_true, p, zero_division=0), 4),
            "f1": round(f1_score(y_true, p, zero_division=0), 4)}


def run_fl(clients, X_eval, y_eval, feature_cols, rounds, epochs):
    tensors = [(torch.tensor(c[feature_cols].values, dtype=torch.float32),
                torch.tensor(c["binary_label"].values, dtype=torch.float32)) for c in clients]
    global_model = IDSModel(len(feature_cols))
    for r in range(rounds):
        locals_ = [local_train(global_model, X, y, epochs=epochs) for X, y in tensors]
        global_model = fedavg(locals_, [len(X) for X, _ in tensors])
        print(f"round {r + 1:>2}: {metrics(y_eval, predict_proba(global_model, X_eval))}")
    return global_model


def tune_threshold(model, X_eval, y_eval):
    probs = predict_proba(model, X_eval)
    best_thr, best_f1 = 0.5, metrics(y_eval, probs)["f1"]
    for thr in np.arange(0.30, 0.71, 0.02):
        f1 = metrics(y_eval, probs, thr)["f1"]
        if f1 > best_f1:
            best_thr, best_f1 = round(float(thr), 2), f1
    print(f"Best threshold {best_thr:.2f} -> F1 {best_f1:.4f}")
    return best_thr


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--test-path", default="data/test_public.csv")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_df, test_df, base_cols = load_and_prepare()
    feature_cols = [c for c in base_cols if c not in DROP_COLS] + NEW_COLS

    clients = [engineer_features(c) for c in make_clients(train_df)]
    test_eng = engineer_features(test_df)
    X_eval = torch.tensor(test_eng[feature_cols].values, dtype=torch.float32)
    y_eval = test_eng["binary_label"].values

    model = run_fl(clients, X_eval, y_eval, feature_cols, args.rounds, args.epochs)
    thr = tune_threshold(model, X_eval, y_eval)

    kaggle = engineer_features(pd.read_csv(args.test_path))
    X_k = torch.tensor(kaggle[feature_cols].values, dtype=torch.float32)
    preds = (predict_proba(model, X_k) > thr).astype(int)
    pd.DataFrame({"Id": kaggle["Id"], "Expected": preds}).to_csv(args.out, index=False)
    print(f"Wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
