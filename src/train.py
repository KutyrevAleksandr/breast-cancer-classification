"""Обучение, подбор гиперпараметров и оценка моделей.

Запуск:  python -m src.train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, окно не нужно
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import load_data
from src.utils import positive_scores


def build_candidates(seed: int) -> dict[str, tuple[Pipeline, dict]]:
    """Модели-кандидаты и сетки гиперпараметров для GridSearchCV.

    Масштабирование стоит ВНУТРИ Pipeline, поэтому при кросс-валидации
    StandardScaler обучается только на train-фолдах (нет утечки данных).
    """
    return {
        "LogisticRegression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, random_state=seed))]),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "SVC": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(random_state=seed))]),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.001]},
        ),
        "RandomForest": (
            Pipeline([("clf", RandomForestClassifier(random_state=seed))]),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10],
             "clf__min_samples_leaf": [1, 3]},
        ),
        "GradientBoosting": (
            Pipeline([("clf", GradientBoostingClassifier(random_state=seed))]),
            {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 3]},
        ),
    }


def evaluate(model, X_test, y_test) -> dict[str, float]:
    """Метрики на отложенной тестовой выборке."""
    scores = positive_scores(model, X_test)
    pred = model.predict(X_test)
    return {
        "roc_auc": roc_auc_score(y_test, scores),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "accuracy": accuracy_score(y_test, pred),
    }


def save_plots(model, X_test, y_test, out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["доброкачественная", "злокачественная"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=names, cmap="Blues", ax=ax)
    ax.set_title("Матрица ошибок (test)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC-кривая (test)")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Permutation importance работает для любой модели
    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                 n_repeats=20, random_state=seed)
    top = (pd.Series(imp.importances_mean, index=X_test.columns)
           .sort_values().tail(10))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    top.plot.barh(ax=ax, color="steelblue")
    ax.set_title("Топ-10 признаков (permutation importance, ΔROC-AUC)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Классификация опухолей молочной железы")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5, help="число фолдов кросс-валидации")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    print(f"Train: {len(X_train)}, test: {len(X_test)}, "
          f"доля злокачественных: {y.mean():.1%}")

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    # Базовая модель: чтобы было с чем сравнивать
    dummy = DummyClassifier(strategy="stratified", random_state=args.seed).fit(X_train, y_train)
    baseline = evaluate(dummy, X_test, y_test)
    print(f"\nBaseline (Dummy)   test ROC-AUC = {baseline['roc_auc']:.3f}")

    # Подбор гиперпараметров для каждой модели
    searches: dict[str, GridSearchCV] = {}
    rows = []
    for name, (pipe, grid) in build_candidates(args.seed).items():
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)
        searches[name] = gs
        rows.append({"model": name, "cv_roc_auc": gs.best_score_,
                     "best_params": gs.best_params_})
        print(f"{name:<18} CV ROC-AUC = {gs.best_score_:.4f}  {gs.best_params_}")

    cv_table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    best_name = cv_table.iloc[0]["model"]
    best_model = searches[best_name].best_estimator_

    # Финальная оценка ОДИН раз на тесте, который не участвовал в подборе
    metrics = evaluate(best_model, X_test, y_test)
    print(f"\nЛучшая модель: {best_name}")
    for k, v in metrics.items():
        print(f"  test {k:<10} = {v:.3f}")

    # Сохраняем результаты
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.models_dir / "best_model.joblib")
    cv_table.to_csv(args.reports_dir / "cv_results.csv", index=False)
    with open(args.reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name,
                   "best_params": searches[best_name].best_params_,
                   "cv_roc_auc": float(cv_table.iloc[0]["cv_roc_auc"]),
                   "test": metrics, "baseline_test": baseline},
                  f, ensure_ascii=False, indent=2)
    save_plots(best_model, X_test, y_test, args.reports_dir / "figures", args.seed)
    print(f"\nМодель -> {args.models_dir}/best_model.joblib, "
          f"отчёты и графики -> {args.reports_dir}/")


if __name__ == "__main__":
    main()
