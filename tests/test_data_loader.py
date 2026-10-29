from io import BytesIO

import pandas as pd
import pytest

from datalens_ai.data_loader import load_csv


def test_load_csv_reads_tabular_data() -> None:
    file = BytesIO(b"name,value\nAlice,3\n")

    result = load_csv(file)

    assert list(result.columns) == ["name", "value"]
    assert result.iloc[0].to_dict() == {"name": "Alice", "value": 3}


def test_load_csv_rejects_empty_file() -> None:
    with pytest.raises(ValueError, match="CSV-файл пуст"):
        load_csv(BytesIO(b""))
