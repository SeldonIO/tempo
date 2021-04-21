from typing import Any

import rclone

from ...conf import settings
from ...utils import logger


def _load_rclone_cfg() -> str:
    with open(settings.rclone_cfg, "r") as f:
        return f.read()


def upload(tempo_artifact: Any):
    "Upload local to remote using rclone"
    model = tempo_artifact.get_tempo()
    local_path = model.details.local_folder
    remote_uri = model.details.uri
    logger.info("Uploading %s to %s", local_path, remote_uri)
    rclone.with_config(_load_rclone_cfg()).copy(local_path, remote_uri, flags=["-P"])


def download(tempo_artifact: Any):
    "Download remote to local using rclone"
    model = tempo_artifact.get_tempo()
    local_path = model.details.local_folder
    remote_uri = model.details.uri
    logger.info("Downloading %s to %s", remote_uri, local_path)
    rclone.with_config(_load_rclone_cfg()).copy(remote_uri, local_path, flags=["-P"])
