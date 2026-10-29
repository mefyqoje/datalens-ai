"""Plotly visualization builders for dataset exploration."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


HISTOGRAM = "Гистограмма"
SCATTER = "Диаграмма рассеяния"
FREQUENCY = "Частоты значений"
CORRELATION = "Тепловая карта корреляций"


def available_visualizations(dataset: pd.DataFrame) -> list[str]:
    """Return chart types supported by the columns in a dataset."""
    numeric_columns = dataset.select_dtypes(include="number").columns
    chart_types = [FREQUENCY]
    if len(numeric_columns) >= 1:
        chart_types.insert(0, HISTOGRAM)
    if len(numeric_columns) >= 2:
        chart_types.extend([SCATTER, CORRELATION])
    return chart_types


def create_visualization(
    dataset: pd.DataFrame,
    chart_type: str,
    column: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
) -> go.Figure:
    """Build a Plotly chart for the selected dataset columns."""
    if chart_type == HISTOGRAM and column is not None:
        return px.histogram(dataset, x=column, title=f"Распределение: {column}")

    if chart_type == FREQUENCY and column is not None:
        values = dataset[column].astype("string").fillna("Пропуск")
        counts = values.value_counts().head(30)
        return px.bar(
            x=counts.index,
            y=counts.values,
            labels={"x": column, "y": "Количество"},
            title=f"Частоты значений: {column} (до 30 значений)",
        )

    if (
        chart_type == SCATTER
        and x_column is not None
        and y_column is not None
    ):
        return px.scatter(
            dataset,
            x=x_column,
            y=y_column,
            title=f"{y_column} от {x_column}",
        )

    if chart_type == CORRELATION:
        numeric_data = dataset.select_dtypes(include="number")
        correlation = numeric_data.corr()
        return px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Корреляции числовых признаков",
        )

    raise ValueError(f"Невозможно построить график типа «{chart_type}».")
