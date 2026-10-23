"""Lab 06: supervised regression mini-projects on reproducible synthetic data."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / 'datasets'; DATA.mkdir(exist_ok=True)
OUT = BASE / 'outputs' / 'lab_06'; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

def compare(frame, target, label, categorical=()):
    frame.to_csv(DATA / f'lab_06_{label}.csv', index=False)
    X, y = frame.drop(columns=[target]), frame[target]
    numeric = [col for col in X.columns if col not in categorical]
    preprocess = ColumnTransformer([
        ('number', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                               ('encode', OneHotEncoder(handle_unknown='ignore'))]), list(categorical))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results, best = [], None
    for name, algo in [('Ridge', Ridge(alpha=1)), ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42))]:
        pipeline = Pipeline([('preprocessor', preprocess), ('estimator', algo)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        metrics = dict(model=name, MAE=mean_absolute_error(y_test, pred),
                       RMSE=np.sqrt(mean_squared_error(y_test, pred)), R2=r2_score(y_test, pred))
        results.append(metrics)
        if best is None or metrics['RMSE'] < best[0]:
            best = (metrics['RMSE'], pipeline, pred, name)
    pd.DataFrame(results).to_csv(OUT / f'{label}_metrics.csv', index=False)
    joblib.dump(best[1], OUT / f'{label}_model.joblib')
    plt.figure(figsize=(5, 4)); plt.scatter(y_test, best[2], alpha=0.5)
    low, high = min(min(y_test), min(best[2])), max(max(y_test), max(best[2]))
    plt.plot([low, high], [low, high], 'k--'); plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.title(f'{label}: {best[3]}'); plt.tight_layout()
    plt.savefig(OUT / f'{label}_predictions.png', dpi=140); plt.close()
    print(label, pd.DataFrame(results).round(3).to_string(index=False))
    print('Example prediction:', round(float(best[1].predict(X.iloc[[0]])[0]), 2))

def main():
    n = 450
    attendance = rng.uniform(45, 100, n)
    internal = rng.uniform(25, 100, n)
    assignments = rng.integers(2, 11, n)
    hours = rng.uniform(.5, 6, n)
    previous = rng.uniform(35, 95, n)
    score = np.clip(.2 * attendance + .24 * internal + 1.6 * assignments + 3 * hours +
                    .25 * previous - 4 + rng.normal(0, 5, n), 0, 100)
    students = pd.DataFrame(dict(attendance=attendance, internal_marks=internal,
                                 assignments_completed=assignments, study_hours=hours,
                                 previous_semester=previous, final_score=score))
    compare(students, 'final_score', 'student_performance')
    area = rng.uniform(500, 3000, n)
    beds = rng.integers(1, 6, n); baths = rng.integers(1, 5, n)
    parking = rng.integers(0, 3, n); age = rng.uniform(0, 35, n)
    location = rng.choice(['urban', 'suburban', 'rural'], n)
    price = (area * 4800 + beds * 190000 + baths * 160000 + parking * 240000 -
             age * 40000 + np.select([location == 'urban', location == 'suburban'],
                                      [3500000, 1500000], default=0) + rng.normal(0, 550000, n))
    houses = pd.DataFrame(dict(area_sqft=area, bedrooms=beds, bathrooms=baths, parking=parking,
                               age_years=age, location=location, price_inr=np.maximum(price, 500000)))
    compare(houses, 'price_inr', 'house_price', categorical=['location'])
    print('Datasets: SYNTHETIC examples only; not actual student or housing records.')

if __name__ == '__main__':
    main()
