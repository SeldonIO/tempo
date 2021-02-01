import pytest
from tempo.serve.loader import _to_rclone
from tempo.serve.model import Model, ModelFramework
from tempo.seldon.docker import SeldonDockerRuntime


@pytest.mark.parametrize(
    "path, expected",
    [
        ("gs://mybucket/sklearn/iris","gcs://mybucket/sklearn/iris"),
        ("s3://mybucket/sklear/iris","s3://mybucket/sklear/iris")
    ]
)
def test_rclone_conversion(path, expected):
    assert _to_rclone(path) == expected
