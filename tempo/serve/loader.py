import cloudpickle
import rclone
import re
import conda_pack
import yaml
import uuid

from typing import Optional
from tempfile import NamedTemporaryFile
from subprocess import run
from tempo.conf import settings
from tempo.serve.constants import MLServerEnvDeps

RCLONE_CONVERSIONS = [("^gs:", "gcs:")]


def save_custom(pipeline, file_path: str) -> str:
    with open(file_path, "wb") as file:
        cloudpickle.dump(pipeline, file)

    return file_path


def load_custom(file_path: str):
    with open(file_path, "rb") as file:
        return cloudpickle.load(file)


def save_environment(file_path: str, env_name: str = None) -> str:
    # TODO: Check if Conda is installed

    env = _get_environment(env_name=env_name)
    env = _add_required_deps(env)

    return _pack_environment(env=env, file_path=file_path)


def _get_environment(env_name: str = None) -> dict:
    cmd = "conda env export"

    if env_name:
        cmd += f" --name {env_name}"

    proc = run(cmd, shell=True, check=True, capture_output=True)
    return yaml.safe_load(proc.stdout)


def _add_required_deps(env: dict) -> dict:
    if "dependencies" not in env:
        env["dependencies"] = []

    dependencies = env["dependencies"]
    pip_deps = _get_pip_deps(dependencies)
    if not pip_deps:
        pip_deps = {"pip": []}
        dependencies.append(pip_deps)

    # TODO: Check if they are present first?
    pip_deps["pip"].extend(MLServerEnvDeps)

    return env


def _get_pip_deps(dependencies: dict) -> Optional[dict]:
    for dep in dependencies:
        if isinstance(dep, dict) and "pip" in dep:
            # If entry is a dict, and has a `pip` key that's the one
            return dep

    return None


def _pack_environment(env: dict, file_path: str) -> str:
    with NamedTemporaryFile(mode="w", suffix='.yml') as file:
        # TODO: Save copy of environment.yaml alongside tarball
        yaml.safe_dump(env, file)

        # Create env
        tmp_env_path = file.name
        tmp_env_name = f"tempo-{uuid.uuid4()}"
        cmd = f"conda env create --name {tmp_env_name} --file {tmp_env_path}"
        run(cmd, shell=True, check=True)

        # Pack environment
        conda_pack.pack(name=tmp_env_name, output=file_path)

        # Remove environment
        cmd = f"conda env remove --name {tmp_env_name}"

    return file_path


def _to_rclone(path: str) -> str:
    "convert standard uris to rclone prefixed ones"
    for p, r in RCLONE_CONVERSIONS:
        path = re.sub(p, r, path)
    return path


def _load_rclone_cfg() -> str:
    with open(settings.rclone_cfg, "r") as f:
        return f.read()


def upload(local_path: str, remote_uri: str):
    "Upload local to remote using rclone"
    remote_uri = _to_rclone(remote_uri)
    rclone.with_config(_load_rclone_cfg()).copy(local_path, remote_uri)


def download(remote_uri: str, local_path: str):
    "Download remote to local using rclone"
    remote_uri = _to_rclone(remote_uri)
    rclone.with_config(_load_rclone_cfg()).copy(remote_uri, local_path)
