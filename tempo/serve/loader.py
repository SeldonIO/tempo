import logging
import re
import uuid
from subprocess import run
from tempfile import NamedTemporaryFile
from typing import Optional

import cloudpickle
import conda_pack
import rclone
import yaml

from tempo.conf import settings
from tempo.serve.constants import MLServerEnvDeps
from tempo.utils import logger


def save_custom(pipeline, file_path: str) -> str:
    with open(file_path, "wb") as file:
        cloudpickle.dump(pipeline, file)

    return file_path


def load_custom(file_path: str):
    with open(file_path, "rb") as file:
        return cloudpickle.load(file)


def _get_env(conda_env_file_path: str = None, env_name: str = None) -> dict:
    if conda_env_file_path:
        with open(conda_env_file_path) as file:
            env = yaml.safe_load(file)
            if not _has_required_deps(env):
                raise ValueError(f"conda.yaml does not contain {MLServerEnvDeps}")
            else:
                logger.info("Using found conda.yaml")
    else:
        env = _get_environment(env_name=env_name)
        env = _add_required_deps(env)
    return env


def save_environment(conda_pack_file_path: str, conda_env_file_path: str = None, env_name: str = None):
    if env_name:
        _pack_environment(env_name, conda_pack_file_path)
    else:
        env = _get_env(conda_env_file_path, env_name)
        _create_and_pack_environment(env=env, file_path=conda_pack_file_path)


def _get_environment(env_name: str = None) -> dict:
    cmd = "conda env export"

    if env_name:
        cmd += f" --name {env_name}"

    proc = run(cmd, shell=True, check=True, capture_output=True)
    return yaml.safe_load(proc.stdout)


def _has_required_deps(env: dict) -> bool:
    if "dependencies" not in env:
        return False

    dependencies = env["dependencies"]
    pip_deps = _get_pip_deps(dependencies)
    if not pip_deps:
        return False

    for dep in MLServerEnvDeps:
        parts = re.split(r"==|>=|<=|~=|!=|>|<|==:", dep)
        module = parts[0]
        r = re.compile(fr"{module}$|({module}((==|>=|<=|~=|!=|>|<|==:)[0-9]+\.[0-9]+.[0-9]+))")
        newlist = list(filter(r.match, pip_deps["pip"]))
        if len(newlist) == 0:
            return False

    return True


def _add_required_deps(env: dict) -> dict:
    if "dependencies" not in env:
        env["dependencies"] = []

    dependencies = env["dependencies"]
    pip_deps = _get_pip_deps(dependencies)
    if not pip_deps:
        pip_deps = {"pip": []}
        dependencies.append(pip_deps)

    for dep in MLServerEnvDeps:
        parts = re.split(r"==|>=|<=|~=|!=|>|<|==:", dep)
        module = parts[0]
        r = re.compile(fr"{module}$|({module}((==|>=|<=|~=|!=|>|<|==:)[0-9]+\.[0-9]+.[0-9]+))")
        newlist = list(filter(r.match, pip_deps["pip"]))
        if len(newlist) == 0:
            pip_deps["pip"].extend(MLServerEnvDeps)

    return env


def _get_pip_deps(dependencies: dict) -> Optional[dict]:
    for dep in dependencies:
        if isinstance(dep, dict) and "pip" in dep:
            # If entry is a dict, and has a `pip` key that's the one
            return dep

    return None


def _pack_environment(env_name: str, file_path: str):
    logger.info("packing conda environment from %s", env_name)
    # Pack environment
    conda_pack.pack(
        name=env_name,
        output=file_path,
        force=True,
        verbose=True,
        ignore_editable_packages=False,
        ignore_missing_files=True,
    )


def _create_and_pack_environment(env: dict, file_path: str):
    with NamedTemporaryFile(mode="w", suffix=".yml") as file:
        # TODO: Save copy of environment.yaml alongside tarball
        yaml.safe_dump(env, file)

        # Create env
        tmp_env_path = file.name
        tmp_env_name = f"tempo-{uuid.uuid4()}"
        cmd = f"conda env create --name {tmp_env_name} --file {tmp_env_path}"
        logger.info("Creating conda env with: %s", cmd)
        run(cmd, shell=True, check=True)

        try:
            _pack_environment(tmp_env_name, file_path)
        finally:
            # Remove environment
            cmd = f"conda remove --name {tmp_env_name} --all --yes"
            logger.info("Removing conda env with: %s", cmd)
            run(cmd, shell=True, check=True)


def _load_rclone_cfg() -> str:
    with open(settings.rclone_cfg, "r") as f:
        return f.read()


def upload(local_path: str, remote_uri: str):
    "Upload local to remote using rclone"
    logger.info("Uploading %s to %s", local_path, remote_uri)
    rclone.with_config(_load_rclone_cfg()).copy(local_path, remote_uri, flags=["-P"])


def download(remote_uri: str, local_path: str):
    "Download remote to local using rclone"
    logger.info("Downloading %s to %s", remote_uri, local_path)
    rclone.with_config(_load_rclone_cfg()).copy(remote_uri, local_path, flags=["-P"])
