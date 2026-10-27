import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from features import SEED, TARGETS, engineer_features, feature_matrix, load_data

st.set_page_config(page_title="Water Pollutants Predictor", page_icon="💧")


@st.cache_resource
def train_model():
    """Trains the best model from train.py (RF, engineered features, log1p targets) on all data."""
    raw = load_data()
    df = engineer_features(raw)
    df = df.dropna(subset=TARGETS + [f"{t}_lag1" for t in TARGETS])
    X_raw = feature_matrix(df)
    imputer = SimpleImputer(strategy="median").fit(X_raw)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    model.fit(imputer.transform(X_raw), np.log1p(df[TARGETS].to_numpy()))
    return raw, model, imputer, list(X_raw.columns)


raw, model, imputer, columns = train_model()

st.title("Water Pollutants Predictor")
st.write(
    "Forecasts the next sample's pollutant levels for a monitoring station from its recent readings. "
    "Model: Random Forest on lag, rolling-mean and seasonal features (test mean R² 0.449 vs 0.300 for the original model)."
)

stations = sorted(raw["id"].unique())
station_id = st.selectbox("Station ID", stations)
history = raw[raw["id"] == station_id]
last_date = history["date"].max()
sample_date = st.date_input("Date of next sample", value=last_date + pd.Timedelta(days=90), min_value=last_date + pd.Timedelta(days=1))

if st.button("Predict"):
    # Append a placeholder row for the new sample so engineer_features builds its lags from the station's history
    new_row = {"id": station_id, "date": pd.Timestamp(sample_date)}
    station_df = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    feats = engineer_features(station_df).tail(1)
    X = feature_matrix(feats).reindex(columns=columns, fill_value=0.0)
    pred = np.expm1(model.predict(imputer.transform(X)))[0]

    st.subheader(f"Predicted levels for station {station_id} on {sample_date}")
    latest = history.sort_values("date").iloc[-1]
    st.dataframe(
        pd.DataFrame({
            "Pollutant": TARGETS,
            "Predicted (mg/L)": np.round(pred, 3),
            f"Last reading {latest['date'].date()}": [latest[t] for t in TARGETS],
        }),
        hide_index=True,
    )
