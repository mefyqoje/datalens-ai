"""Dataset overview and data-quality summaries."""

import pandas as pd


def profile_dataset(dataset: pd.DataFrame) -> dict[str, object]:
    """Return basic dataset statistics and per-column details."""
    columns_info = pd.DataFrame(
        {
            "Столбец": dataset.columns,
            "Тип": dataset.dtypes.astype(str).values,
            "Пропуски": dataset.isna().sum().values,
            "Уникальные значения": dataset.nunique(dropna=True).values,
        }
    )
    summary = (
        dataset.describe(include="all")
        .transpose()
        .rename_axis("Столбец")
        .reset_index()
    )
    return {
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "duplicates": int(dataset.duplicated().sum()),
        "missing_values": int(dataset.isna().sum().sum()),
        "columns_info": columns_info,
        "summary": summary,
    }
