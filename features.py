"""Data loading and feature engineering shared by train.py and app.py."""
import numpy as np
import pandas as pd

DATA_PATH = "PB_All_2000_2021.csv"
TARGETS = ["O2", "NO3", "NO2", "SO4", "PO4", "CL"]
MEASURED = ["NH4", "BSK5", "Suspended", "O2", "NO3", "NO2", "SO4", "PO4", "CL"]
FEATURE_COLS = [f"{c}_{s}" for c in MEASURED for s in ("lag1", "roll3")] + ["days_since_prev", "year", "month_sin", "month_cos"]
SEED = 42


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    return df.sort_values(["id", "date"]).reset_index(drop=True)


def engineer_features(df):
    """Per-station history features. Only past samples are used, so there is no leakage."""
    df = df.copy()
    g = df.groupby("id")
    for col in MEASURED:
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_roll3"] = g[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["days_since_prev"] = g["date"].diff().dt.days
    df["year"] = df["date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    return df


def station_dummies(df):
    return pd.get_dummies(df["id"].astype(str), prefix="st", dtype=float)


def feature_matrix(df):
    """Station one-hot + engineered history features (output of engineer_features)."""
    return pd.concat([station_dummies(df), df[FEATURE_COLS]], axis=1)
