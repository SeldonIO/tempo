import os

import pytest

from tempo.serve.constants import DefaultEnvFilename, MLServerEnvDeps
from tempo.serve.loader.env import (
    _add_required_deps,
    _create_and_pack_environment,
    _get_env,
    _get_environment,
    _get_pip_deps,
    _has_required_deps,
    save_environment,
)


@pytest.mark.skip(reason="very slow")
def test_save_environment(tmp_path):
    env_path = os.path.join(tmp_path, DefaultEnvFilename)
    save_environment(conda_pack_file_path=env_path, env_name="base")

    assert os.path.isfile(env_path)


def test_create_and_pack_environment(tmp_path):
    env = {"dependencies": [{"pip": MLServerEnvDeps}]}
    env_path = os.path.join(tmp_path, DefaultEnvFilename)

    _create_and_pack_environment(env=env, file_path=env_path)

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
        {
            "foo": "bar",
            "dependencies": ["foo==1.0", {"pip": ["foo-pip==1.0"]}],
        },
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
        ({}, False),
        (
            {
                "foo": "bar",
                "dependencies": ["foo==1.0", {"pip": ["mlserver"]}],
            },
            True,
        ),
    ],
)
def test_has_required_deps(env, expected):
    has_deps = _has_required_deps(env)
    assert has_deps == expected


@pytest.mark.parametrize(
    "env_file_path",
    [
        os.path.join(os.path.dirname(__file__), "data", "conda.yaml"),
    ],
)
def test_get_env_ok(env_file_path):
    _get_env(conda_env_file_path=env_file_path)


@pytest.mark.parametrize(
    "env_file_path",
    [
        os.path.join(os.path.dirname(__file__), "data", "conda_missing_mlserver.yaml"),
    ],
)
def test_get_env_not_ok(env_file_path):
    with pytest.raises(ValueError):
        _get_env(conda_env_file_path=env_file_path)
