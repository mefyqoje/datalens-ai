"""CSV input loading and validation."""

from typing import BinaryIO

import pandas as pd


def load_csv(file: BinaryIO) -> pd.DataFrame:
    """Read a CSV file and reject empty datasets."""
    try:
        dataset = pd.read_csv(file)
    except pd.errors.EmptyDataError as error:
        raise ValueError("CSV-файл пуст или не содержит заголовков.") from error

    if dataset.empty and len(dataset.columns) == 0:
        raise ValueError("CSV-файл не содержит столбцов.")

    return dataset
