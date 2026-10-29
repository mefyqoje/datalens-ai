import pandas as pd
import pytest

from datalens_ai.modeling import TaskType, train_and_evaluate


def test_classification_trains_and_compares_baseline() -> None:
    dataset = pd.DataFrame(
        {
            "number": range(20),
            "category": ["a", "b"] * 10,
            "target": ["low", "high"] * 10,
        }
    )

    result = train_and_evaluate(dataset, "target", "classification")

    assert result["task_type"] == "classification"
    assert result["train_rows"] + result["test_rows"] == len(dataset)
    assert set(result["metrics"]) == {"accuracy", "f1_weighted"}
    assert set(result["baseline_metrics"]) == {"accuracy", "f1_weighted"}
    assert set(result["model_metrics"]) == {
        "linear",
        "decision_tree",
        "random_forest",
    }


def test_regression_handles_missing_features_and_compares_baseline() -> None:
    dataset = pd.DataFrame(
        {
            "number": [1, None, 3, 4, 5, 6, 7, 8, 9, 10],
            "category": ["a", "b"] * 5,
            "target": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
    )

    result = train_and_evaluate(dataset, "target", "regression")

    assert result["task_type"] == "regression"
    assert set(result["metrics"]) == {"mae", "rmse", "r2"}
    assert set(result["baseline_metrics"]) == {"mae", "rmse", "r2"}
    assert set(result["model_metrics"]) == {
        "linear",
        "decision_tree",
        "random_forest",
    }


def test_training_can_run_only_selected_algorithms() -> None:
    dataset = pd.DataFrame(
        {"feature": range(12), "target": [value * 3 for value in range(12)]}
    )

    result = train_and_evaluate(
        dataset,
        "target",
        "regression",
        algorithms=["linear", "decision_tree"],
    )

    assert result["algorithms"] == ["linear", "decision_tree"]
    assert set(result["model_metrics"]) == {"linear", "decision_tree"}


def test_training_rejects_empty_or_unknown_algorithm_selection() -> None:
    dataset = pd.DataFrame(
        {"feature": range(8), "target": range(8)}
    )

    with pytest.raises(ValueError):
        train_and_evaluate(dataset, "target", "regression", algorithms=[])
    with pytest.raises(ValueError):
        train_and_evaluate(
            dataset, "target", "regression", algorithms=["unknown"]
        )


@pytest.mark.parametrize(
    ("dataset", "target", "task_type", "error"),
    [
        (
            pd.DataFrame({"feature": [1, 2, 3, 4], "target": [1, 1, 1, 1]}),
            "target",
            "classification",
            "минимум 2 класса",
        ),
        (
            pd.DataFrame({"feature": [1, 2, 3, 4], "target": ["a", "b", "a", "b"]}),
            "target",
            "regression",
            "должен быть числовым",
        ),
        (
            pd.DataFrame({"feature": [1, 2, 3, 4], "target": [1, 2, 3, 4]}),
            "missing",
            "regression",
            "отсутствует",
        ),
    ],
)
def test_training_rejects_invalid_inputs(
    dataset: pd.DataFrame, target: str, task_type: TaskType, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        train_and_evaluate(dataset, target, task_type)
