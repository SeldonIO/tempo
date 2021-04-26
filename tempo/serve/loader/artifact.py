import os
from typing import Any

import cloudpickle

from ..constants import DefaultModelFilename, DefaultRemoteFilename


def save(tempo_artifact: Any, save_env=True):
    model = tempo_artifact.get_tempo()
    model.save(save_env=save_env)


def save_custom(pipeline, file_path: str) -> str:
    with open(file_path, "wb") as file:
        cloudpickle.dump(pipeline, file)

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
