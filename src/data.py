"""Загрузка данных.

Используется датасет Wisconsin Diagnostic Breast Cancer (WDBC),
который встроен в scikit-learn, поэтому скачивать ничего не нужно.
569 пациенток, 30 числовых признаков, извлечённых из изображений
тонкоигольной аспирационной биопсии (радиус, текстура, периметр и т. д.).
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Возвращает признаки X и целевую переменную y.

    В оригинале в sklearn 0 = злокачественная, 1 = доброкачественная.
    Переворачиваем, чтобы положительный класс (1) был «злокачественная
    опухоль»: тогда recall и precision отвечают на медицински важный
    вопрос «сколько злокачественных опухолей мы нашли».
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    y = (1 - raw.target).rename("malignant")
    return X, y


if __name__ == "__main__":
    X, y = load_data()
    print(f"Объектов: {X.shape[0]}, признаков: {X.shape[1]}")
    print("Пропусков:", int(X.isna().sum().sum()))
    print("Баланс классов:")
    print(y.value_counts(normalize=True).rename({0: "доброкачественная", 1: "злокачественная"}).round(3))
