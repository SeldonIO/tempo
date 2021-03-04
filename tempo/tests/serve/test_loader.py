import pytest
import os

from tempo.serve.loader import (
    save_environment,
    _to_rclone,
    _get_environment,
    _add_required_deps,
    _get_pip_deps,
    _pack_environment,
    _has_required_deps,
    _get_env
)
from tempo.serve.constants import MLServerEnvDeps, DefaultEnvFilename


@pytest.mark.parametrize(
    "path, expected",
    [
        ("gs://mybucket/sklearn/iris", "gcs://mybucket/sklearn/iris"),
        ("s3://mybucket/sklear/iris", "s3://mybucket/sklear/iris"),
    ],
)
def test_rclone_conversion(path, expected):
    assert _to_rclone(path) == expected

@pytest.mark.skip(reason="very slow")
def test_save_environment(tmp_path):
    env_path = os.path.join(tmp_path, DefaultEnvFilename)
    save_environment(conda_pack_file_path=env_path, env_name='base')

    assert os.path.isfile(env_path)


def test_pack_environment(tmp_path):
    env = {"dependencies": [{"pip": MLServerEnvDeps}]}

    env_path = os.path.join(tmp_path, DefaultEnvFilename)
    _pack_environment(env=env, file_path=env_path)

    assert os.path.isfile(env_path)


def test_get_environment():
    env = _get_environment(env_name="base")

    assert env["name"] == "base"


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"foo": "bar"},
        {"foo": "bar", "dependencies": ["foo==1.0"]},
        {"foo": "bar", "dependencies": ["foo==1.0", {"pip": ["foo-pip==1.0"]}],},
    ],
)
def test_add_required_deps(env):
    env_with_deps = _add_required_deps(env=env)

    pip_deps = _get_pip_deps(env_with_deps["dependencies"])
    for dep in MLServerEnvDeps:
        assert dep in pip_deps["pip"]


@pytest.mark.parametrize(
    "env, expected",
    [
        ({},False),
        ({"foo": "bar", "dependencies": ["foo==1.0", {"pip": ["mlserver"]}], }, False),
        ({"foo": "bar", "dependencies": ["foo==1.0", {"pip": ["mlserver","mlserver-tempo"]}], }, True),
    ],
)
def test_has_required_deps(env, expected):
    has_deps = _has_required_deps(env)
    assert has_deps == expected

@pytest.mark.parametrize(
    "env_file_path",
    [
       os.path.join(os.path.dirname(__file__), 'data', 'conda.yaml'),
    ],
)
def test_get_env_ok(env_file_path):
    _get_env(conda_env_file_path=env_file_path)

@pytest.mark.parametrize(
    "env_file_path",
    [
       os.path.join(os.path.dirname(__file__), 'data', 'conda_missing_mlserver.yaml'),
    ],
)
def test_get_env_ok(env_file_path):
    with pytest.raises(ValueError):
        _get_env(conda_env_file_path=env_file_path)