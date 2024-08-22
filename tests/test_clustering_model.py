from pathlib import Path
from pandas.testing import assert_frame_equal
from ml_api.cluster_model import clustering_data, read_data


def test_clustering_data() -> None:
    _path = Path(Path(__file__).resolve().parent / "data/german_credit_data_testing.csv")
    _data = read_data(_path)
    _prediction_data = _data.drop("Cluster", axis=1)
    _prediction_data = clustering_data(_prediction_data)
    assert_frame_equal(_data, _prediction_data)
