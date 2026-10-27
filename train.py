"""Reproducible training + evaluation pipeline for water-quality prediction.

Compares, on the same time-based split (train < 2018, test >= 2018):
  1. Persistence baseline  - predict each station's previous reading
  2. Random Forest, original features (station one-hot + year), as in the notebook
  3. Random Forest, engineered features (lags, rolling means, seasonality, gap days)
  3b. Same Random Forest trained on log1p targets
  4. PyTorch MLP, engineered features (log1p targets)

Every model is logged to MLflow (sqlite:///mlflow.db; view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`). Results are also written to results.json.

Usage: python train.py
"""
import json

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from features import SEED, TARGETS, engineer_features, feature_matrix, load_data, station_dummies

TEST_START = "2018-01-01"


def evaluate(y_true, y_pred):
    per_target = {}
    for i, t in enumerate(TARGETS):
        per_target[t] = {
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
        }
    mean_r2 = float(np.mean([m["r2"] for m in per_target.values()]))
    return {"mean_r2": mean_r2, "per_target": per_target}


class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, X_val, y_val, epochs=400, patience=30):
    """Trains on log1p-scaled, standardized targets with early stopping on a time-based validation set."""
    torch.manual_seed(SEED)
    x_scaler, y_scaler = StandardScaler().fit(X_tr), StandardScaler().fit(np.log1p(y_tr))
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xt, yt = to_t(x_scaler.transform(X_tr)), to_t(y_scaler.transform(np.log1p(y_tr)))
    Xv, yv = to_t(x_scaler.transform(X_val)), to_t(y_scaler.transform(np.log1p(y_val)))

    model = MLP(Xt.shape[1], yt.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_loss, best_state, bad_epochs, best_epoch = float("inf"), None, 0, 0
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt)).split(64):
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, bad_epochs, best_epoch = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0, epoch
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)

    def predict(X):
        model.eval()
        with torch.no_grad():
            out = model(to_t(x_scaler.transform(X))).numpy()
        return np.expm1(y_scaler.inverse_transform(out))

    return predict, best_epoch


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("water-quality")

    df = engineer_features(load_data())
    # Need a previous reading per station and all six targets present
    df = df.dropna(subset=TARGETS + [f"{t}_lag1" for t in TARGETS]).reset_index(drop=True)
    is_test = df["date"] >= TEST_START
    y = df[TARGETS].to_numpy()
    y_train, y_test = y[~is_test], y[is_test]
    print(f"Rows: {len(df)} | train (<2018): {(~is_test).sum()} | test (>=2018): {is_test.sum()} | stations: {df['id'].nunique()}")

    results = {}

    # 1. Persistence baseline
    pred = df.loc[is_test, [f"{t}_lag1" for t in TARGETS]].to_numpy()
    results["persistence_baseline"] = evaluate(y_test, pred)

    # 2. Random Forest, original notebook features
    X_orig = pd.concat([station_dummies(df), df[["year"]]], axis=1).to_numpy()
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_orig[~is_test], y_train)
    results["rf_original_features"] = evaluate(y_test, rf.predict(X_orig[is_test]))

    # 3. Random Forest, engineered features
    X_eng_raw = feature_matrix(df)
    imputer = SimpleImputer(strategy="median").fit(X_eng_raw[~is_test])
    X_eng = imputer.transform(X_eng_raw)
    rf_eng = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1).fit(X_eng[~is_test], y_train)
    results["rf_engineered_features"] = evaluate(y_test, rf_eng.predict(X_eng[is_test]))

    # 3b. Same RF trained on log1p targets (the pollutant distributions are heavily right-skewed)
    rf_log = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1).fit(X_eng[~is_test], np.log1p(y_train))
    results["rf_engineered_features_log_target"] = evaluate(y_test, np.expm1(rf_log.predict(X_eng[is_test])))

    importances = pd.Series(rf_log.feature_importances_, index=X_eng_raw.columns).sort_values(ascending=False)

    # 4. PyTorch MLP, engineered features; last training year is the early-stopping validation set
    train_idx = np.where(~is_test)[0]
    is_val = df.loc[train_idx, "date"] >= "2017-01-01"
    tr, val = train_idx[~is_val.to_numpy()], train_idx[is_val.to_numpy()]
    predict_mlp, best_epoch = train_mlp(X_eng[tr], y[tr], X_eng[val], y[val])
    results["pytorch_mlp_engineered_features"] = evaluate(y_test, predict_mlp(X_eng[is_test]))
    results["pytorch_mlp_engineered_features"]["best_epoch"] = best_epoch

    for name, res in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"test_start": TEST_START, "n_train": int((~is_test).sum()), "n_test": int(is_test.sum())})
            mlflow.log_metric("mean_r2", res["mean_r2"])
            for t, m in res["per_target"].items():
                mlflow.log_metric(f"r2_{t}", m["r2"])
                mlflow.log_metric(f"mae_{t}", m["mae"])

    print(f"\n{'model':34s} {'mean R2':>8s}  " + "  ".join(f"{t:>7s}" for t in TARGETS))
    for name, res in results.items():
        print(f"{name:34s} {res['mean_r2']:8.3f}  " + "  ".join(f"{res['per_target'][t]['r2']:7.3f}" for t in TARGETS))
    print("\nTop 10 features (RF engineered, log target):")
    print(importances.head(10).round(3).to_string())

    results["top_features_rf_log_target"] = importances.head(10).round(4).to_dict()
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
