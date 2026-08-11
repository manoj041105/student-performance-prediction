"""
Flask web interface for the Student Performance Prediction System.

Loads the trained model (models/student_performance_model.joblib) and serves
a form where you enter a student's details to get a predicted final score.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.

If no trained model exists yet, run student_performance_prediction.py first
to generate one.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

MODEL_PATH = "models/student_performance_model.joblib"

app = Flask(__name__)

_model = None


def get_model():
    """Loads the trained model once and caches it."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                "Run `python student_performance_prediction.py` first to train and save one."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def engineer_features(row: dict) -> pd.DataFrame:
    """Mirrors the feature engineering done in student_performance_prediction.py."""
    study_engagement_index = row["study_hours_per_week"] * (row["attendance_rate"] / 100)
    sleep_deviation = abs(row["sleep_hours"] - 7.5)

    return pd.DataFrame([{
        "study_hours_per_week": row["study_hours_per_week"],
        "attendance_rate": row["attendance_rate"],
        "previous_gpa": row["previous_gpa"],
        "sleep_hours": row["sleep_hours"],
        "study_engagement_index": study_engagement_index,
        "sleep_deviation": sleep_deviation,
        "parental_support": row["parental_support"],
        "extracurricular": row["extracurricular"],
        "part_time_job": row["part_time_job"],
        "gender": row["gender"],
    }])


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_values = {
        "study_hours_per_week": 10,
        "attendance_rate": 85,
        "previous_gpa": 3.0,
        "sleep_hours": 7,
        "parental_support": "Medium",
        "extracurricular": "No",
        "part_time_job": "No",
        "gender": "Female",
    }

    if request.method == "POST":
        try:
            form_values.update({
                "study_hours_per_week": float(request.form["study_hours_per_week"]),
                "attendance_rate": float(request.form["attendance_rate"]),
                "previous_gpa": float(request.form["previous_gpa"]),
                "sleep_hours": float(request.form["sleep_hours"]),
                "parental_support": request.form["parental_support"],
                "extracurricular": request.form["extracurricular"],
                "part_time_job": request.form["part_time_job"],
                "gender": request.form["gender"],
            })

            model = get_model()
            X = engineer_features(form_values)
            pred = model.predict(X)[0]
            prediction = round(float(pred), 1)
        except FileNotFoundError as e:
            error = str(e)
        except Exception as e:
            error = f"Couldn't generate a prediction: {e}"

    return render_template("index.html", prediction=prediction, error=error, values=form_values)


if __name__ == "__main__":
    app.run(debug=True)
