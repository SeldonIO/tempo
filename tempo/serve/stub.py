import json
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import urlparse

import rclone

from tempo.serve.base import BaseModel, DeployedModel, ModelSpec

from ..conf import settings
from ..utils import logger


def _load_rclone_cfg() -> str:
    with open(settings.rclone_cfg, "r") as f:
        return f.read()


def deserialize(d: dict) -> DeployedModel:
    ms = ModelSpec(**d)
    return DeployedModel(ms)


def save_remote(model: BaseModel, uri: str):
    if uri.startswith(".") or uri.startswith("/"):
        logger.info("Writing remote to %s", uri)
        with open(uri, "w") as f:
            f.write(model.serialize())
    else:
        logger.info("Uploading to %s", uri)
        with NamedTemporaryFile(mode="w", suffix=".yml") as file:
            file.write(model.serialize())
            file.close()
            rclone.with_config(_load_rclone_cfg()).copy(file.name, uri, flags=["-P"])


def load_remote(uri: str) -> DeployedModel:
    if uri.startswith(".") or uri.startswith("/"):
        with open(uri, "r") as f:
            return deserialize(json.loads(f.read()))
    else:
        with TemporaryDirectory() as folder:
            rclone.with_config(_load_rclone_cfg()).copy(uri, folder, flags=["-P"])
            a = urlparse(uri)
            filename = folder + os.path.basename(a.path)
            with open(filename, "r") as f:
                return deserialize(json.loads(f.read()))
