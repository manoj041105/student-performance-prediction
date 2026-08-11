# Student Performance Prediction System

A machine learning regression system that predicts student academic performance
from historical educational data.

**Stack:** Python, Pandas, NumPy, Scikit-learn

## What it does

- Built a machine learning regression model to predict student academic
  performance using historical educational data.
- Performed data preprocessing, feature engineering, and model evaluation
  to improve prediction accuracy.

## Pipeline

1. **Data loading** — reads `data/student_data.csv` if present; otherwise
   generates a realistic synthetic dataset of 1,000 student records
   (study hours, attendance, GPA, sleep, parental support, extracurriculars,
   part-time job, gender) and saves it there.
2. **Preprocessing** — median imputation + scaling for numeric features,
   most-frequent imputation + one-hot encoding for categorical features,
   assembled with `sklearn.compose.ColumnTransformer`.
3. **Feature engineering** — derives `study_engagement_index` (study hours
   weighted by attendance) and `sleep_deviation` (distance from the
   recommended 7.5 hours of sleep).
4. **Modeling** — trains and compares `LinearRegression` and
   `RandomForestRegressor`, then tunes the Random Forest with
   `GridSearchCV` (n_estimators, max_depth).
5. **Evaluation** — reports MAE, RMSE, and R² for every candidate model and
   keeps the one with the lowest RMSE.
6. **Outputs** — saves the best model to `models/student_performance_model.joblib`
   and a predicted-vs-actual scatter plot to `outputs/prediction_vs_actual.png`.

## Web interface

Once you have a trained model (`models/student_performance_model.joblib`),
you can run a local Flask app to enter a student's details in a browser
and get a predicted score:

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Getting started

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python student_performance_prediction.py
```

## Using your own data

Drop a CSV at `data/student_data.csv` with these columns and it will be used
automatically instead of the synthetic dataset:

```
study_hours_per_week, attendance_rate, previous_gpa, sleep_hours,
parental_support, extracurricular, part_time_job, gender, final_score
```

## Project structure

```
student-performance-prediction/
├── student_performance_prediction.py   # main pipeline
├── requirements.txt
├── data/                               # generated/raw dataset (gitignored)
├── models/                             # saved trained model (gitignored)
├── outputs/                            # evaluation plots (gitignored)
└── README.md
```

## Sample results (synthetic data)

| Model                | MAE  | RMSE | R²    |
|-----------------------|------|------|-------|
| Linear Regression      | ~3.8 | ~4.6 | ~0.59 |
| Random Forest (tuned)  | ~3.5 | ~4.6 | ~0.59 |

Results will vary slightly by run/seed and are far more meaningful once a
real historical dataset is substituted in.
