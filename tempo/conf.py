from pydantic import BaseSettings
from pathlib import Path


class TempoSettings(BaseSettings):
    rclone_cfg: str = str(Path.home()) + "/.config/rclone/rclone.conf"


settings = TempoSettings()
