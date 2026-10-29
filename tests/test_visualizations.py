import pandas as pd
import pytest

from datalens_ai.visualizations import (
    CORRELATION,
    FREQUENCY,
    HISTOGRAM,
    SCATTER,
    available_visualizations,
    create_visualization,
)


def test_available_visualizations_depend_on_numeric_columns() -> None:
    categorical = pd.DataFrame({"label": ["a", "b"]})
    mixed = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    numeric = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    assert available_visualizations(categorical) == [FREQUENCY]
    assert available_visualizations(mixed) == [HISTOGRAM, FREQUENCY]
    assert available_visualizations(numeric) == [
        HISTOGRAM,
        FREQUENCY,
        SCATTER,
        CORRELATION,
    ]


@pytest.mark.parametrize(
    ("chart_type", "kwargs"),
    [
        (HISTOGRAM, {"column": "x"}),
        (FREQUENCY, {"column": "label"}),
        (SCATTER, {"x_column": "x", "y_column": "y"}),
        (CORRELATION, {}),
    ],
)
def test_create_visualization_returns_plotly_figure(
    chart_type: str, kwargs: dict[str, str]
) -> None:
    dataset = pd.DataFrame(
        {"x": [1, 2, 3], "y": [3, 2, 1], "label": ["a", "b", "a"]}
    )

    figure = create_visualization(dataset, chart_type, **kwargs)

    assert figure.data


def test_create_visualization_rejects_missing_column() -> None:
    with pytest.raises(ValueError, match="Невозможно построить"):
        create_visualization(pd.DataFrame({"x": [1]}), HISTOGRAM)
