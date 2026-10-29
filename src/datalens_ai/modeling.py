"""Baseline classification and regression workflows."""

import math
from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

TaskType = Literal["classification", "regression"]

ALGORITHM_LABELS = {
    "linear": "Линейная модель",
    "decision_tree": "Дерево решений",
    "random_forest": "Случайный лес",
}


def available_algorithms(task_type: TaskType) -> list[str]:
    """Return algorithm identifiers supported for a task type."""
    if task_type not in ("classification", "regression"):
        raise ValueError("Тип задачи должен быть classification или regression.")
    return list(ALGORITHM_LABELS)


def train_and_evaluate(
    dataset: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    test_size: float = 0.2,
    random_state: int = 42,
    algorithms: list[str] | None = None,
) -> dict[str, object]:
    """Train selected models and compare them with a dummy baseline."""
    if target_column not in dataset.columns:
        raise ValueError("Выбранный целевой столбец отсутствует в данных.")
    if not 0 < test_size < 1:
        raise ValueError("Размер тестовой выборки должен быть между 0 и 1.")
    selected_algorithms = (
        available_algorithms(task_type) if algorithms is None else algorithms
    )
    if not selected_algorithms:
        raise ValueError("Выберите хотя бы один алгоритм.")
    unknown_algorithms = set(selected_algorithms) - set(ALGORITHM_LABELS)
    if unknown_algorithms:
        raise ValueError(
            "Неизвестные алгоритмы: " + ", ".join(sorted(unknown_algorithms))
        )

    labeled_data = dataset.dropna(subset=[target_column])
    if len(labeled_data) < 4:
        raise ValueError("Для обучения и проверки нужно минимум 4 строки с целевым значением.")

    features = labeled_data.drop(columns=[target_column]).copy()
    target = labeled_data[target_column]
    if features.empty:
        raise ValueError("Для обучения нужна хотя бы одна колонка-признак.")
    if features.isna().all().any():
        empty_columns = features.columns[features.isna().all()].tolist()
        raise ValueError(
            "Нельзя использовать полностью пустые признаки: "
            + ", ".join(map(str, empty_columns))
        )

    if task_type == "classification":
        if target.nunique() < 2:
            raise ValueError("Для классификации целевой столбец должен содержать минимум 2 класса.")
        class_counts = target.value_counts()
        test_count = math.ceil(len(target) * test_size)
        train_count = len(target) - test_count
        if (
            class_counts.min() < 2
            or test_count < len(class_counts)
            or train_count < len(class_counts)
        ):
            raise ValueError(
                "Для классификации нужно минимум по 2 строки на класс и достаточно "
                "строк для каждого класса в обучающей и тестовой выборках."
            )
        stratify = target
    elif task_type == "regression":
        if not pd.api.types.is_numeric_dtype(target):
            raise ValueError("Для регрессии целевой столбец должен быть числовым.")
        stratify = None
    else:
        raise ValueError("Тип задачи должен быть classification или regression.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    numeric_columns = x_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in x_train.columns if column not in numeric_columns
    ]
    for column in categorical_columns:
        if pd.api.types.is_datetime64_any_dtype(x_train[column]):
            x_train[column] = x_train[column].astype("string")
            x_test[column] = x_test[column].astype("string")

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocessing = ColumnTransformer(transformers=transformers)

    if task_type == "classification":
        baseline = DummyClassifier(strategy="most_frequent")
    else:
        baseline = DummyRegressor(strategy="mean")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)
    if task_type == "classification":
        baseline_metrics = {
            "accuracy": float(accuracy_score(y_test, baseline_predictions)),
            "f1_weighted": float(
                f1_score(
                    y_test, baseline_predictions, average="weighted", zero_division=0
                )
            ),
        }
    else:
        baseline_metrics = {
            "mae": float(mean_absolute_error(y_test, baseline_predictions)),
            "rmse": float(math.sqrt(mean_squared_error(y_test, baseline_predictions))),
            "r2": float(r2_score(y_test, baseline_predictions)),
        }

    model_metrics: dict[str, dict[str, float]] = {}
    for algorithm in selected_algorithms:
        if task_type == "classification":
            estimators = {
                "linear": LogisticRegression(max_iter=1000),
                "decision_tree": DecisionTreeClassifier(random_state=random_state),
                "random_forest": RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            }
        else:
            estimators = {
                "linear": Ridge(),
                "decision_tree": DecisionTreeRegressor(random_state=random_state),
                "random_forest": RandomForestRegressor(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            }
        model = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("model", estimators[algorithm]),
            ]
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "f1_weighted": float(
                    f1_score(y_test, predictions, average="weighted", zero_division=0)
                ),
            }
        else:
            metrics = {
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(math.sqrt(mean_squared_error(y_test, predictions))),
                "r2": float(r2_score(y_test, predictions)),
            }
        model_metrics[algorithm] = metrics

    return {
        "task_type": task_type,
        "target_column": target_column,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "algorithms": selected_algorithms,
        "model_metrics": model_metrics,
        "metrics": model_metrics[selected_algorithms[0]],
        "baseline_metrics": baseline_metrics,
    }
