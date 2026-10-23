"""Lab 10 capstone mini-apps: loan approval and symptom-pattern classification.

All datasets generated here are SYNTHETIC. Neither predictor is suitable for
loan underwriting, medical diagnosis, or use with real personal information.
Run `python lab_10.py` to train, validate, and persist both models.
"""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from generate_labs_09_10_data import make_loans, make_symptoms

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs' / 'lab_10'
OUTPUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def _train_task(task: str, data: pd.DataFrame, target: str,
                numeric_features: list[str], categoric_features: list[str],
                *, binary: bool) -> dict:
    """Evaluate two algorithms on a held-out 20% split; save best by weighted F1."""
    feature_columns = numeric_features + categoric_features
    X, y = data[feature_columns], data[target]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=SEED)
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]), numeric_features),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categoric_features)
    ], remainder='drop')
    algorithms = {
        'LogisticRegression': LogisticRegression(max_iter=1500, random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=160,
                                               min_samples_leaf=2, random_state=SEED)
    }
    comparisons = []
    best_model, best_name, best_score = None, '', -1
    for name, estimator in algorithms.items():
        # A new pipeline per candidate: no preprocessing fitted on the test split.
        pipeline = Pipeline([('preprocessing', preprocess), ('classifier', estimator)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metric = {
            'model': name,
            'accuracy': float(accuracy_score(y_test, predictions)),
            'precision_weighted': float(precision_score(y_test, predictions,
                                                         average='weighted',zero_division=0)),
            'recall_weighted': float(recall_score(y_test, predictions,
                                                  average='weighted',zero_division=0)),
            'f1_weighted': float(f1_score(y_test, predictions,average='weighted'))
        }
        if binary:
            positive_idx = list(pipeline.classes_).index(1)
            probability = pipeline.predict_proba(x_test)[:, positive_idx]
            metric['roc_auc'] = float(roc_auc_score(y_test, probability))
        comparisons.append(metric)
        if metric['f1_weighted'] > best_score:
            best_score = metric['f1_weighted']
            best_name = name
            best_model = pipeline
    if best_model is None:
        raise RuntimeError('No successfully trained model')
    best_predictions = best_model.predict(x_test)
    pd.DataFrame(comparisons).to_csv(OUTPUT / f'{task}_model_comparison.csv',index=False)
    pd.DataFrame(classification_report(y_test,best_predictions,
                  output_dict=True,zero_division=0)).transpose().to_csv(
                      OUTPUT / f'{task}_classification_report.csv')
    joblib.dump(best_model, OUTPUT / f'{task}_pipeline.joblib')
    fig, ax = plt.subplots(figsize=(7,5))
    ConfusionMatrixDisplay.from_predictions(y_test,best_predictions,
                                            ax=ax,xticks_rotation=25,cmap='Blues')
    ax.set_title(f'{task.replace("_", " ").title()} — {best_name}')
    fig.tight_layout()
    fig.savefig(OUTPUT / f'{task}_confusion_matrix.png',dpi=150)
    plt.close(fig)
    summary = {'dataset_rows': len(data),'train_rows': len(x_train),
               'test_rows':len(x_test),'best_model': best_name,
               'best_weighted_f1':best_score,'candidate_metrics':comparisons,
               'synthetic_data':True,
               'disclaimer': ('Educational simulated exercise only. Metrics do not imply '
                              'creditworthiness or medical validity.')}
    (OUTPUT / f'{task}_metrics.json').write_text(json.dumps(summary,indent=2))
    example_pred = best_model.predict(x_test.iloc[[0]])[0]
    print(f'{task}: best={best_name}, held-out weighted F1={best_score:.4f},'
          f' example predicted label={example_pred}, actual={y_test.iloc[0]}')
    return summary


def train_loan() -> dict:
    path = ROOT / 'datasets' / 'lab_10_loan_SYNTHETIC.csv'
    if not path.exists():
        make_loans()
    frame = pd.read_csv(path)
    return _train_task('loan_approval', frame, 'approved',
                       numeric_features=['income_monthly_inr','loan_amount_inr',
                                         'credit_history','loan_term_months'],
                       categoric_features=['employment','education','marital_status'],
                       binary=True)


def train_conditions() -> dict:
    path = ROOT / 'datasets' / 'lab_10_symptoms_SYNTHETIC.csv'
    if not path.exists():
        make_symptoms()
    frame = pd.read_csv(path)
    features = [col for col in frame if col != 'condition_label']
    return _train_task('symptom_pattern',frame,'condition_label',
                       numeric_features=features,categoric_features=[],binary=False)


def main():
    train_loan()
    train_conditions()
    print('Saved pipelines and evaluation charts under outputs/lab_10')
    print('WARNING: never use either demonstration model for real-world decisions.')


if __name__ == '__main__':
    main()
