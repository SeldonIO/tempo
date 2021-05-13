import os

from tempo.conf import TempoSettings

TESTS_PATH = os.path.dirname(__file__)
TESTDATA_PATH = os.path.join(TESTS_PATH, "testdata")


def test_env():
    env_path = os.path.join(TESTDATA_PATH, "tempo.env")
    settings = TempoSettings(_env_file=env_path)
    assert settings.use_kubernetes
