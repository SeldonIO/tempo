import os
import sys
from typing import Any

import cloudpickle

from ..constants import DefaultModelFilename, DefaultRemoteFilename


def save(tempo_artifact: Any, save_env=True):
    model = tempo_artifact.get_tempo()
    model.save(save_env=save_env)


def save_custom(pipeline, module: str, file_path: str) -> str:
    if module is not None:
        modules_to_capture = [module]
    else:
        modules_to_capture = []

    # Hack to force cloudpickle to capture the whole function instead of just referencing the
    # code file.
    # See https://github.com/cloudpipe/cloudpickle/blob/74d69d759185edaeeac7bdcb7015cfc0c652f204/
    # cloudpickle/cloudpickle.py#L490
    old_modules = {}
    try:  # Try is needed to restore the state if something goes wrong
        for module_name in modules_to_capture:
            if module_name in sys.modules:
                old_modules[module_name] = sys.modules.pop(module_name)
        with open(file_path, "wb") as file:
            cloudpickle.dump(pipeline, file)
    finally:
        sys.modules.update(old_modules)

    return file_path


def load(folder: str):
    file_path_pkl = os.path.join(folder, DefaultModelFilename)
    return load_custom(file_path_pkl)


def load_custom(file_path: str):
    with open(file_path, "rb") as file:
        return cloudpickle.load(file)


def load_remote(folder: str):
    file_path = os.path.join(folder, DefaultRemoteFilename)
    with open(file_path, "rb") as file:
        return cloudpickle.load(file)
