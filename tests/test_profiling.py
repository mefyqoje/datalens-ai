import pandas as pd

from datalens_ai.profiling import profile_dataset


def test_profile_dataset_reports_dimensions_and_quality() -> None:
    dataset = pd.DataFrame({"value": [1, 1, None], "label": ["a", "a", "b"]})

    profile = profile_dataset(dataset)

    assert profile["rows"] == 3
    assert profile["columns"] == 2
    assert profile["duplicates"] == 1
    assert profile["missing_values"] == 1
    assert profile["columns_info"]["Пропуски"].tolist() == [1, 0]
    assert "mean" in profile["summary"].columns
