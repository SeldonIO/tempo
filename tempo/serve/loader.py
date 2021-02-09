import cloudpickle
import rclone
import re
from tempo.conf import settings


def save_custom(pipeline, file_path: str) -> str:
    with open(file_path, "wb") as file:
        cloudpickle.dump(pipeline, file)

    return file_path


def load_custom(file_path: str):
    with open(file_path, "rb") as file:
        return cloudpickle.load(file)


RCLONE_CONVERSIONS = [("^gs:", "gcs:")]


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
