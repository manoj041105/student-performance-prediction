"""
Student Performance Prediction System
--------------------------------------
Builds a machine learning regression model to predict student academic
performance (final exam score) from historical educational data.

Pipeline:
    1. Load / generate historical student data
    2. Data preprocessing (missing values, encoding, scaling)
    3. Feature engineering (derived features)
    4. Train/test split + model training (Linear Regression, Random Forest)
    5. Model evaluation (MAE, RMSE, R^2) + feature importance
    6. Save the trained model and a prediction-vs-actual plot

Tech stack: Python, Pandas, NumPy, Scikit-learn
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
DATA_PATH = "data/student_data.csv"
MODEL_PATH = "models/student_performance_model.joblib"
PLOT_PATH = "outputs/prediction_vs_actual.png"


# --------------------------------------------------------------------------- #
# 1. Data loading / generation
# --------------------------------------------------------------------------- #
def generate_synthetic_dataset(n_students: int = 1000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset of historical student records.
    If you have a real dataset (e.g. the UCI "Student Performance" dataset),
    drop it at DATA_PATH with matching columns and load_data() will use it
    instead of generating synthetic data.
    """
    rng = np.random.default_rng(seed)

    study_hours_per_week = rng.normal(10, 4, n_students).clip(0, 30)
    attendance_rate = rng.normal(85, 10, n_students).clip(40, 100)
    previous_gpa = rng.normal(3.0, 0.5, n_students).clip(1.0, 4.0)
    sleep_hours = rng.normal(7, 1.3, n_students).clip(3, 10)
    parental_support = rng.choice(["Low", "Medium", "High"], n_students, p=[0.25, 0.5, 0.25])
    extracurricular = rng.choice(["Yes", "No"], n_students, p=[0.4, 0.6])
    part_time_job = rng.choice(["Yes", "No"], n_students, p=[0.3, 0.7])
    gender = rng.choice(["Male", "Female"], n_students)

    # introduce a few missing values to make preprocessing meaningful
    missing_idx = rng.choice(n_students, size=int(n_students * 0.03), replace=False)
    attendance_rate = attendance_rate.astype(float)
    attendance_rate[missing_idx] = np.nan

    support_bonus = {"Low": -3, "Medium": 0, "High": 3}
    job_penalty = {"Yes": -2, "No": 0}
    extra_bonus = {"Yes": 1.5, "No": 0}

    noise = rng.normal(0, 5, n_students)
    attendance_filled = np.nan_to_num(attendance_rate, nan=float(np.nanmean(attendance_rate)))

    final_score = (
        40
        + study_hours_per_week * 1.8
        + attendance_filled * 0.25
        + previous_gpa * 6
        - np.abs(sleep_hours - 7.5) * 1.2
        + np.array([support_bonus[s] for s in parental_support])
        + np.array([job_penalty[s] for s in part_time_job])
        + np.array([extra_bonus[s] for s in extracurricular])
        + noise
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame({
        "study_hours_per_week": study_hours_per_week.round(1),
        "attendance_rate": attendance_rate.round(1),
        "previous_gpa": previous_gpa.round(2),
        "sleep_hours": sleep_hours.round(1),
        "parental_support": parental_support,
        "extracurricular": extracurricular,
        "part_time_job": part_time_job,
        "gender": gender,
        "final_score": final_score.round(1),
    })
    return df


def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        print(f"Loading dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("No dataset found at data/student_data.csv — generating a synthetic dataset instead.")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df = generate_synthetic_dataset()
    df.to_csv(DATA_PATH, index=False)
    return df


# --------------------------------------------------------------------------- #
# 2 & 3. Preprocessing + feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Derived feature: study efficiency = study hours weighted by attendance
    df["study_engagement_index"] = df["study_hours_per_week"] * (df["attendance_rate"].fillna(df["attendance_rate"].mean()) / 100)
    # Derived feature: sleep deviation from the recommended 7.5h
    df["sleep_deviation"] = (df["sleep_hours"] - 7.5).abs()
    return df


def build_preprocessing_pipeline(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


# --------------------------------------------------------------------------- #
# 4 & 5. Training + evaluation
# --------------------------------------------------------------------------- #
def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n[{name}] MAE={mae:.2f}  RMSE={rmse:.2f}  R^2={r2:.3f}")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2, "preds": preds}


def main():
    df = load_data()
    df = engineer_features(df)

    target = "final_score"
    numeric_features = ["study_hours_per_week", "attendance_rate", "previous_gpa",
                         "sleep_hours", "study_engagement_index", "sleep_deviation"]
    categorical_features = ["parental_support", "extracurricular", "part_time_job", "gender"]

    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

    candidates = {
        "LinearRegression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]),
        "RandomForest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=RANDOM_STATE)),
        ]),
    }

    results = []
    fitted_models = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        results.append(evaluate(pipeline, X_test, y_test, name))

    # Light hyperparameter search on the Random Forest model
    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [None, 8, 12],
    }
    grid = GridSearchCV(candidates["RandomForest"], param_grid, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    fitted_models["RandomForest (tuned)"] = best_rf
    results.append(evaluate(best_rf, X_test, y_test, "RandomForest (tuned)"))
    print(f"Best RF params: {grid.best_params_}")

    best_result = min(results, key=lambda r: r["rmse"])
    best_model = fitted_models[best_result["model"]]
    print(f"\nBest model: {best_result['model']} (lowest RMSE)")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_result["preds"], alpha=0.5)
    lims = [0, 100]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual final score")
    plt.ylabel("Predicted final score")
    plt.title(f"{best_result['model']}: Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved prediction plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
