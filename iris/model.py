import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import optuna

import numpy as np
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path(__file__).parent.parent / "data"
PARAMS_FILE = DATA_DIR / "best_params.json"

def load_and_train():
    df = pd.read_csv(DATA_DIR / "Training.csv")
    df.columns = [c.replace(" ", "_") for c in df.columns]

    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]
    le = LabelEncoder()
    y_le = le.fit_transform(y)
    columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y_le, test_size=0.3, random_state=42)

    if PARAMS_FILE.exists():
        params = json.loads(PARAMS_FILE.read_text())
        print("Loaded best parameters from file.")
    else:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            }
            model = XGBClassifier(**params, eval_metric="mlogloss", random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
            return model.score(X_test, y_test)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        params = study.best_params
        PARAMS_FILE.write_text(json.dumps(params, indent=2))
        print("Best parameters found and saved to file.")
        print(f"Val accuracy: {study.best_value:.4f}")

    model = XGBClassifier(**params, eval_metric="mlogloss", random_state=42, n_jobs=-1)
    model.fit(X, y_le)

    test_dataset = pd.read_csv(DATA_DIR / "Testing.csv")
    test_dataset.columns = [c.replace(" ", "_") for c in test_dataset.columns]
    X_test_final = test_dataset.drop("prognosis", axis=1)
    y_test_final = test_dataset["prognosis"]
    y_test_final_le = le.transform(y_test_final)
    test_accuracy = model.score(X_test_final, y_test_final_le)
    print(f"Test accuracy: {test_accuracy:.4f}")
    return model, columns, le

def predict_top3(model, cols, text, le):
    row = np.zeros(len(cols))
    for i, symptoms in enumerate(cols):
        if symptoms in text.lower():
            row[i] = 1

    proba = model.predict_proba(pd.DataFrame([row], columns=cols))[0]
    top3_idx = proba.argsort()[-3:][::-1]
    return [{"disease": le.inverse_transform([idx])[0], "confidence": round(float(proba[idx]) * 100, 1)} for idx in top3_idx]
