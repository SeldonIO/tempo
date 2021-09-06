import os

import pytest

from tempo import ModelFramework
from tempo.serve.constants import DefaultEnvFilename
from tempo.serve.loader.env import (
    _add_required_deps,
    _create_and_pack_environment,
    _get_env,
    _get_environment,
    _get_mlserver_deps,
    _get_pip_deps,
    _has_required_deps,
    _is_dep_not_defined,
    save_environment,
)


@pytest.mark.skip(reason="very slow")
def test_save_environment(tmp_path):
    env_path = os.path.join(tmp_path, DefaultEnvFilename)
    save_environment(conda_pack_file_path=env_path, env_name="base")

    assert os.path.isfile(env_path)


@pytest.mark.parametrize("platform", [None, ModelFramework.MLFlow])
def test_create_and_pack_environment(tmp_path, platform):
    env = {"dependencies": [{"pip": _get_mlserver_deps(platform)}]}
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
@pytest.mark.parametrize("platform", [None, ModelFramework.MLFlow])
def test_add_required_deps(env, platform):
    env_with_deps = _add_required_deps(env=env, platform=platform)

    pip_deps = _get_pip_deps(env_with_deps["dependencies"])
    for dep in _get_mlserver_deps(platform):
        assert not _is_dep_not_defined(dep, pip_deps["pip"])


@pytest.mark.parametrize(
    "env, platform, expected",
    [
        ({}, None, False),
        (
            {
                "foo": "bar",
                "dependencies": ["foo==1.0", {"pip": ["mlserver"]}],
            },
            None,
            True,
        ),
        (
            {
                "foo": "bar",
                "dependencies": ["foo==1.0", {"pip": ["mlserver"]}],
            },
            ModelFramework.MLFlow,
            False,
        ),
        (
            {
                "foo": "bar",
                "dependencies": ["foo==1.0", {"pip": ["mlserver", "mlserver-dummy"]}],
            },
            ModelFramework.MLFlow,
            False,
        ),
        (
            {
                "foo": "bar",
                "dependencies": ["foo==1.0", {"pip": ["mlserver", "mlserver-mlflow"]}],
            },
            ModelFramework.MLFlow,
            True,
        ),
        (
            {
                "foo": "bar",
                # old version is still accepted
                "dependencies": ["foo==1.0", {"pip": ["mlserver", "mlserver-mlflow==0.1.0"]}],
            },
            ModelFramework.MLFlow,
            True,
        ),
    ],
)
def test_has_required_deps(env, platform, expected):
    has_deps = _has_required_deps(env, platform)
    assert has_deps == expected


@pytest.mark.parametrize(
    "env_file_path",
    [
        os.path.join(os.path.dirname(__file__), "data", "conda.yaml"),
    ],
)
def test_get_env_ok(env_file_path):
    _get_env(conda_env_file_path=env_file_path)


@pytest.mark.parametrize("platform", ModelFramework)
def test_get_env_with_missing_mlserver_deps(conda_yaml_no_mlserver_deps, platform):
    env = _get_env(conda_env_file_path=conda_yaml_no_mlserver_deps, platform=platform)
    assert _has_required_deps(env, platform=platform)
