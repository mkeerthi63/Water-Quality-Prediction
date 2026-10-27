# Water Quality Prediction

Predicts six water-pollutant levels (O2, NO3, NO2, SO4, PO4, CL) for river monitoring stations, so that rising pollution can be flagged before the next lab sample. Started during a one-month **AICTE Virtual Internship sponsored by Shell** (June 2025) and extended afterwards with feature engineering, a deep-learning comparison, experiment tracking and a deployable app.

---

## Dataset

`PB_All_2000_2021.csv`: 2,861 lab samples from 22 monitoring stations, 2000–2021, with 9 measured parameters per sample (NH4, BSK5/BOD5, Suspended solids, O2, NO3, NO2, SO4, PO4, CL). Parameter descriptions are in `Parameters_WQM_RMS-min.pdf`.

## Approach

**Time-based split.** Models are trained on samples before 2018 and tested on 2018–2021 (2,429 train / 294 test rows), so every score measures prediction of *future* samples.

**Feature engineering** (`features.py`). The original model used only station ID and year. Error analysis showed that this cannot explain O2 or NO2. Added, per station and using only past samples:
- previous reading of all 9 parameters (`*_lag1`)
- 3-sample rolling mean of all 9 parameters (`*_roll3`)
- days since the previous sample
- month as sine/cosine (seasonality)

**Models compared** (`train.py`), all on the same split:

| Model | Mean R² | O2 | NO3 | NO2 | SO4 | PO4 | CL |
|---|---|---|---|---|---|---|---|
| Persistence baseline (repeat last reading) | -0.818 | -0.098 | -0.018 | -5.654 | 0.783 | -0.642 | 0.719 |
| Random Forest, original features (station + year) | 0.300 | -0.083 | 0.009 | 0.141 | 0.850 | 0.009 | 0.877 |
| Random Forest, engineered features | 0.049 | 0.178 | -0.937 | -0.754 | 0.889 | -0.029 | 0.949 |
| **Random Forest, engineered features, log1p targets** | **0.449** | **0.263** | 0.372 | 0.144 | 0.859 | **0.104** | **0.955** |
| PyTorch MLP (128-64, dropout, early stopping), log1p targets | 0.408 | 0.166 | **0.373** | **0.181** | 0.857 | 0.057 | 0.816 |

Findings:
- Engineered features plus a log1p target transform raised mean R² from **0.300 to 0.449** over the original model, with gains on O2, NO3, PO4 and CL.
- Without the log transform, the Random Forest on engineered features overfit the heavy right-skewed outliers in NO3 and NO2 (R² below 0).
- The PyTorch MLP was competitive (0.408) and best on NO3 and NO2, but the Random Forest was better overall.
- Most important features: 3-sample rolling means of CL and SO4, then year and seasonality.
- NO2 and PO4 remain hard to predict (R² ≤ 0.18); these would need upstream or weather data.

Note: the log-target Random Forest variant was added after seeing the first results, so its test score is slightly optimistic. The test set is small (294 rows).

## Experiment tracking

Each model is logged to **MLflow** (parameters plus per-target R² and MAE). Results are also saved to `results.json`.

```bash
pip install -r requirements-train.txt
python train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Streamlit app

`app.py` forecasts the next sample for a chosen station and date from that station's recent readings, using the best model above. It trains on startup (a few seconds), so no model file needs to be downloaded.

```bash
pip install -r requirements.txt
streamlit run app.py
```

To deploy on Streamlit Community Cloud: New app → this repository → main file `app.py`.

## Files

| File | Purpose |
|---|---|
| `features.py` | Data loading and feature engineering |
| `train.py` | Model comparison, evaluation and MLflow logging |
| `app.py` | Streamlit prediction app |
| `Water_Quality_Prediction.ipynb` | Original internship notebook (exploration and first model) |
| `results.json` | Latest evaluation results |

## Internship Details

- **Internship Type**: AICTE Virtual Internship - Edunet Foundation
- **Sponsor**: Shell
- **Duration**: June 2025 (1 month)
- **Focus Area**: Machine Learning in Environmental Monitoring
