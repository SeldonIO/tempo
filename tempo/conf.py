from pathlib import Path

from pydantic import BaseSettings


class TempoSettings(BaseSettings):
    rclone_cfg: str = str(Path.home()) + "/.config/rclone/rclone.conf"


settings = TempoSettings()
