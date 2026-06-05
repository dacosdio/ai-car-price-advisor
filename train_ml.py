"""
AI Car Price Advisor - Optimized Machine Learning Training
==========================================================
Iteration 1: Linear Regression vs Random Forest (baseline comparison)
Iteration 2: Random Forest (tuned) vs Gradient Boosting vs HistGradientBoosting

Key improvements over the initial version:
- No data leakage: target encoding learned inside the Pipeline (only on training folds)
- Log-target variants tested for right-skewed price distribution
- More engineered features: log_mileage_km, interaction terms
- Model selected by CV R2, not test R2
- Comprehensive evaluation exports
"""

from __future__ import annotations

import json
import pickle
import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "cars_project.csv"
MODEL_PATH    = BASE_DIR / "car_price_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"
LEADERBOARD_PATH   = BASE_DIR / "model_leaderboard.csv"
WORST_ERRORS_PATH  = BASE_DIR / "worst_predictions.csv"

CURRENT_YEAR = 2024
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
TARGET_ENCODING_SMOOTHING = 20.0
TOP_K_WORST_ERRORS = 50

PRICE_MIN = 1_000
PRICE_MAX = 300_000

FUEL_MAPPING = {
    "Super 95": "Petrol", "Super Plus E10 98": "Petrol",
    "Super E10 95": "Petrol", "Regular/Benzine 91": "Petrol",
    "Regular/Benzine E10 91": "Petrol", "Super Plus 98": "Petrol",
    "Diesel": "Diesel", "Biodiesel": "Diesel",
    "Electricity": "Electric",
    "Liquid petroleum gas (LPG)": "Other",
    "Domestic gas H": "Other", "Vegetable oil": "Other",
}

BODY_TYPES    = ["Coupe", "Hatchback", "Sedan", "SUV"]
TRANSMISSIONS = ["Automatic", "Manual", "Semi-automatic"]
FUEL_CATEGORIES = ["Petrol", "Diesel", "Electric", "Other"]


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def extract_first_number(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"([-+]?\d+(?:[\.,]\d+)?)", expand=False)
    return pd.to_numeric(extracted.str.replace(",", ".", regex=False), errors="coerce")

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def serializable(result: Dict) -> Dict:
    return {k: v for k, v in result.items() if k not in {"estimator", "y_pred"}}


# ─── FEATURE ENGINEERING (no leakage) ─────────────────────────────────────────
class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature Engineering:
      - car_age            = CURRENT_YEAR - registration_year
      - mileage_per_year   = mileage_km / car_age
      - log_mileage_km     = log(1 + mileage_km)
      - age_x_mileage      = car_age * mileage_km  (interaction)
      - model_price_encoded = smoothed target encoding per model name
                              (learned ONLY on training data — no leakage)
      - fuel_category      = mapped from raw fuel string
    """

    def __init__(self, current_year: int = CURRENT_YEAR,
                 smoothing: float = TARGET_ENCODING_SMOOTHING) -> None:
        self.current_year = current_year
        self.smoothing    = smoothing

    def fit(self, X: pd.DataFrame, y=None) -> "CarFeatureEngineer":
        if y is None:
            self.global_mean_    = 50_000.0
            self.model_target_map_ = {}
        else:
            y_s = pd.Series(y, index=X.index, dtype=float)
            self.global_mean_ = float(y_s.mean())
            stats = (
                pd.DataFrame({"model": X["model"].fillna("Unknown"), "price": y_s})
                .groupby("model")["price"]
                .agg(["mean", "count"])
            )
            smoothed = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) \
                       / (stats["count"] + self.smoothing)
            self.model_target_map_ = smoothed.to_dict()

        years = pd.to_datetime(X["registration_date"], errors="coerce").dt.year
        fallback = years.dropna().median()
        self.fallback_year_ = int(fallback) if not pd.isna(fallback) else self.current_year - 8
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in ["make", "model", "body_type", "transmission", "primary_fuel"]:
            if col in X.columns:
                X[col] = X[col].fillna("Unknown").astype(str)

        reg_year = pd.to_datetime(X["registration_date"], errors="coerce").dt.year \
                     .fillna(self.fallback_year_).astype(int)
        X["car_age"]          = (self.current_year - reg_year).clip(lower=1)
        X["mileage_km"]       = extract_first_number(X["mileage_km"]).fillna(0)
        X["mileage_per_year"] = X["mileage_km"] / X["car_age"].replace(0, 1)
        X["log_mileage_km"]   = np.log1p(X["mileage_km"].clip(lower=0))
        X["age_x_mileage"]    = X["car_age"] * X["mileage_km"]
        X["fuel_category"]    = X["primary_fuel"].map(FUEL_MAPPING).fillna("Other")
        X["model_price_encoded"] = (
            X["model"].map(self.model_target_map_)
            .fillna(getattr(self, "global_mean_", 50_000.0))
        )
        return X


# ─── DATA ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("AI Car Price Advisor — Optimized ML Training")
print("=" * 70)
print(f"\nLoading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df["price"] = extract_first_number(df["price"])
df = df.dropna(subset=["price"])
df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
df = df[df["body_type"].isin(BODY_TYPES)].copy()
for col in ["make", "model", "transmission", "primary_fuel", "registration_date"]:
    df[col] = df[col].fillna("Unknown")

print(f"After cleaning: {df.shape}")
print(f"Price range: {df['price'].min():,.0f} – {df['price'].max():,.0f} EUR")
print(f"Unique models: {df['model'].nunique()}")

RAW_FEATURES = ["make", "model", "mileage_km", "registration_date",
                "body_type", "transmission", "primary_fuel"]
NUMERIC_FEATURES = ["mileage_km", "log_mileage_km", "car_age",
                    "mileage_per_year", "model_price_encoded", "age_x_mileage"]
CATEGORICAL_FEATURES = ["make", "body_type", "transmission", "fuel_category"]

X = df[RAW_FEATURES].copy()
y = df["price"].astype(float).copy()

# Price-stratified split
try:
    bins = pd.qcut(y, q=10, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=bins)
    print("Split: price-stratified")
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print("Split: random")

print(f"Train: {len(X_train)} | Test: {len(X_test)}")


# ─── PIPELINE FACTORY ──────────────────────────────────────────────────────────
def make_pipeline(model, log_target: bool = False, scale: bool = False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    prep = ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe()),
        ]), CATEGORICAL_FEATURES),
    ], remainder="drop")

    pipe = Pipeline([
        ("engineer", CarFeatureEngineer()),
        ("prep", prep),
        ("model", model),
    ])

    if log_target:
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    return pipe


# ─── EVALUATION ────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    errors = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    ape    = errors / np.maximum(np.abs(y_true), 1)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_abs_error": float(np.median(errors)),
        "within_5k":  float(np.mean(errors <= 5_000)),
        "within_10k": float(np.mean(errors <= 10_000)),
        "within_20k": float(np.mean(errors <= 20_000)),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "within_20pct": float(np.mean(ape <= 0.20)),
    }

def evaluate(name, estimator, X_tr, y_tr, X_te, y_te):
    print(f"\n  {name}")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    try:
        cv_res = cross_validate(estimator, X_tr, y_tr, cv=cv,
                                scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
                                error_score=np.nan)
        fitted = clone(estimator)
        fitted.fit(X_tr, y_tr)
        y_pred = np.maximum(fitted.predict(X_te), 0)
        test   = metrics(y_te, y_pred)
        cv_r2  = float(np.nanmean(cv_res["test_r2"]))
        cv_std = float(np.nanstd(cv_res["test_r2"]))
        cv_mae = float(-np.nanmean(cv_res["test_mae"]))
        print(f"    CV R²: {cv_r2:.4f} ± {cv_std:.4f}  |  Test R²: {test['r2']:.4f}  |  "
              f"MAE: {test['mae']:,.0f}  |  RMSE: {test['rmse']:,.0f}  |  "
              f"±10k: {test['within_10k']:.1%}")
        return {"name": name, "cv_r2_mean": cv_r2, "cv_r2_std": cv_std,
                "cv_mae": cv_mae, **test, "estimator": fitted, "y_pred": y_pred}
    except Exception as e:
        print(f"    Skipped: {e}")
        return None


# ─── ITERATION 1: Linear Regression vs Random Forest ──────────────────────────
print("\n" + "=" * 70)
print("Iteration 1 — Baseline: Linear Regression vs Random Forest")
print("=" * 70)

results = []
for name, model, log_t, scale in [
    ("Linear Regression (raw)",     LinearRegression(),  False, True),
    ("Linear Regression (log)",     LinearRegression(),  True,  True),
    ("Random Forest n=100 (raw)",   RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False, False),
    ("Random Forest n=100 (log)",   RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), True,  False),
]:
    r = evaluate(name, make_pipeline(model, log_t, scale), X_train, y_train, X_test, y_test)
    if r:
        results.append(r)

# ─── ITERATION 2: Tuned RF vs GB vs HistGB ─────────────────────────────────────
print("\n" + "=" * 70)
print("Iteration 2 — Tuned models: RF, GradientBoosting, HistGradientBoosting")
print("=" * 70)

for name, model, log_t in [
    ("Random Forest tuned (raw)",
     RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2,
                           random_state=RANDOM_STATE, n_jobs=-1), False),
    ("Random Forest tuned (log)",
     RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2,
                           random_state=RANDOM_STATE, n_jobs=-1), True),
    ("Gradient Boosting (raw)",
     GradientBoostingRegressor(n_estimators=350, learning_rate=0.05,
                               max_depth=4, subsample=0.85, random_state=RANDOM_STATE), False),
    ("Gradient Boosting (log)",
     GradientBoostingRegressor(n_estimators=350, learning_rate=0.05,
                               max_depth=4, subsample=0.85, random_state=RANDOM_STATE), True),
    ("HistGradientBoosting (raw)",
     HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                   l2_regularization=0.05, random_state=RANDOM_STATE), False),
    ("HistGradientBoosting (log)",
     HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                   l2_regularization=0.05, random_state=RANDOM_STATE), True),
]:
    r = evaluate(name, make_pipeline(model, log_t), X_train, y_train, X_test, y_test)
    if r:
        results.append(r)

# ─── SELECT BEST ───────────────────────────────────────────────────────────────
results.sort(key=lambda r: r["cv_r2_mean"], reverse=True)
best = results[0]

print("\n" + "=" * 70)
print(f"Best Model (by CV R²): {best['name']}")
print(f"  CV R²  : {best['cv_r2_mean']:.4f} ± {best['cv_r2_std']:.4f}")
print(f"  Test R²: {best['r2']:.4f}")
print(f"  RMSE   : {best['rmse']:,.0f} EUR")
print(f"  MAE    : {best['mae']:,.0f} EUR")
print(f"  Median error: {best['median_abs_error']:,.0f} EUR")
print(f"  Within ±5k : {best['within_5k']:.1%}")
print(f"  Within ±10k: {best['within_10k']:.1%}")
print(f"  Within ±20k: {best['within_20k']:.1%}")
print(f"  Within ±10%: {best['within_10pct']:.1%}")
print(f"  Within ±20%: {best['within_20pct']:.1%}")
print("=" * 70)

# ─── PLOTS ─────────────────────────────────────────────────────────────────────
lb = pd.DataFrame([serializable(r) for r in results])
lb.to_csv(LEADERBOARD_PATH, index=False)

top_plot = lb.head(10)
fig, ax = plt.subplots(figsize=(12, max(4, len(top_plot) * 0.4)))
ax.barh(top_plot["name"][::-1], top_plot["cv_r2_mean"][::-1])
ax.set_xlabel("CV R² mean")
ax.set_title("Model Comparison — CV R² (higher = better)")
plt.tight_layout()
plt.savefig(BASE_DIR / "model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

y_pred = best["y_pred"]
lim = max(float(y_test.max()), float(np.max(y_pred)))
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.3, s=18, c="#4878CF")
ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Price (EUR)")
ax.set_ylabel("Predicted Price (EUR)")
ax.set_title(f"Predicted vs Actual — {best['name']}")
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "predicted_vs_actual.png", dpi=120, bbox_inches="tight")
plt.close()

residuals = np.asarray(y_pred) - np.asarray(y_test)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred, residuals, alpha=0.3, s=18, c="#D65F5F")
ax.axhline(0, linestyle="--", lw=1.5)
ax.set_xlabel("Predicted Price (EUR)")
ax.set_ylabel("Residual (EUR)")
ax.set_title(f"Residuals — {best['name']}")
plt.tight_layout()
plt.savefig(BASE_DIR / "residuals.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: model_comparison.png, predicted_vs_actual.png, residuals.png")

# Worst predictions
errors_arr = np.abs(np.asarray(y_pred) - np.asarray(y_test))
worst = X_test.copy().reset_index(drop=True)
worst["actual_price"]    = y_test.reset_index(drop=True)
worst["predicted_price"] = np.round(y_pred, 0).astype(int)
worst["abs_error"]       = np.round(errors_arr, 0).astype(int)
worst.sort_values("abs_error", ascending=False).head(TOP_K_WORST_ERRORS).to_csv(WORST_ERRORS_PATH, index=False)

# ─── SAVE ──────────────────────────────────────────────────────────────────────
makes = sorted(df["make"].dropna().unique().tolist())
metadata = {
    "best_model": best["name"],
    "current_year": CURRENT_YEAR,
    "body_types": BODY_TYPES,
    "transmissions": TRANSMISSIONS,
    "fuel_categories": FUEL_CATEGORIES,
    "fuel_mapping": FUEL_MAPPING,
    "makes": makes,
    "raw_features": RAW_FEATURES,
    "r2": best["r2"],
    "rmse": best["rmse"],
    "mae": best["mae"],
    "within_10k": best["within_10k"],
    "leaderboard_top10": lb.head(10).to_dict(orient="records"),
}

artifact = {"pipeline": best["estimator"], "metadata": metadata}

with MODEL_PATH.open("wb") as f:
    pickle.dump(artifact, f)
print(f"Model saved: {MODEL_PATH}")

with METADATA_PATH.open("w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"Metadata saved: {METADATA_PATH}")
print(f"Leaderboard: {LEADERBOARD_PATH}")
print("\nDone!")
