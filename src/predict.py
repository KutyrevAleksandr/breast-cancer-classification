"""Предсказание обученной моделью.

Без аргументов берёт 5 первых объектов датасета для демонстрации:
    python -m src.predict
Со своим CSV (те же 30 столбцов, что в датасете WDBC):
    python -m src.predict --csv my_data.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data import load_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/best_model.joblib"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit("Модель не найдена. Сначала запустите: python -m src.train")
    model = joblib.load(args.model)

    if args.csv:
        X = pd.read_csv(args.csv)
    else:
        X, _ = load_data()
        X = X.head(5)

    labels = model.predict(X)
    result = pd.DataFrame({
        "prediction": ["злокачественная" if l == 1 else "доброкачественная" for l in labels],
    })
    if hasattr(model, "predict_proba"):
        result.insert(0, "p_malignant", model.predict_proba(X)[:, 1].round(3))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
