import re
import uuid
from subprocess import run
from tempfile import NamedTemporaryFile
from typing import List, Optional

import conda_pack
import yaml

from ...utils import logger
from ..constants import MLServerEnvDeps, MLServerRuntimeEnvDeps
from ..metadata import ModelFramework


def _get_env(conda_env_file_path: str = None, env_name: str = None, platform: ModelFramework = None) -> dict:
    if conda_env_file_path:
        with open(conda_env_file_path) as file:
            logger.info(f"Using found conda env: {conda_env_file_path}")
            env = yaml.safe_load(file)
            env = _add_required_deps_if_missing(env, platform)
    else:
        env = _get_environment(env_name=env_name)
        env = _add_required_deps_if_missing(env, platform)
    return env


def _add_required_deps_if_missing(env: dict, platform: Optional[ModelFramework]) -> dict:
    if not _has_required_deps(env, platform):
        logger.info(f"conda.yaml does not contain {MLServerEnvDeps}, adding them")
        env = _add_required_deps(env, platform)
    return env


def save_environment(
    conda_pack_file_path: str, conda_env_file_path: str = None, env_name: str = None, platform: ModelFramework = None
) -> None:
    if env_name:
        # TODO: add mlserver deps here if not present?
        _pack_environment(env_name, conda_pack_file_path)
    else:
        env = _get_env(conda_env_file_path, env_name, platform)
        _create_and_pack_environment(env=env, file_path=conda_pack_file_path)


def _get_environment(env_name: str = None) -> dict:
    cmd = "conda env export"

    if env_name:
        cmd += f" --name {env_name}"

    proc = run(cmd, shell=True, check=True, capture_output=True)
    return yaml.safe_load(proc.stdout)


def _has_required_deps(env: dict, platform: ModelFramework = None) -> bool:
    if "dependencies" not in env:
        return False

    dependencies = env["dependencies"]
    pip_deps = _get_pip_deps(dependencies)
    if not pip_deps:
        return False

    deps = _get_mlserver_deps(platform)

    for dep in deps:
        if _is_dep_not_defined(dep, pip_deps["pip"]):
            return False

    return True


def _get_mlserver_deps(platform: Optional[ModelFramework]) -> List[str]:
    runtime_deps = MLServerRuntimeEnvDeps.get(platform)  # type: ignore
    deps = MLServerEnvDeps
    if runtime_deps:
        return deps + runtime_deps
    return deps


def _add_required_deps(env: dict, platform: ModelFramework = None) -> dict:
    if "dependencies" not in env:
        env["dependencies"] = []

    dependencies = env["dependencies"]
    pip_deps = _get_pip_deps(dependencies)
    if not pip_deps:
        pip_deps = {"pip": []}
        dependencies.append(pip_deps)

    deps = _get_mlserver_deps(platform)

    for dep in deps:
        if _is_dep_not_defined(dep, pip_deps["pip"]):
            pip_deps["pip"].append(dep)

    return env


def _is_dep_not_defined(dep: str, deps: List[str]) -> bool:
    parts = re.split(r"==|>=|<=|~=|!=|>|<|==:", dep)
    module = parts[0]
    r = re.compile(fr"{module}$|({module}((==|>=|<=|~=|!=|>|<|==:)[0-9]+\.[0-9]+.[0-9]+))")
    newlist = list(filter(r.match, deps))
    return len(newlist) == 0


def _get_pip_deps(dependencies: dict) -> Optional[dict]:
    for dep in dependencies:
        if isinstance(dep, dict) and "pip" in dep:
            # If entry is a dict, and has a `pip` key that's the one
            return dep

    return None


def _pack_environment(env_name: str, file_path: str):
    logger.info(f"packing conda environment from {env_name} to {file_path}")
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
