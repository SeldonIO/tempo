from pydantic import BaseModel
from enum import Enum
from typing import Optional, List, Type


class ModelFramework(Enum):
    SKLearn = "sklearn"
    XGBoost = "xgboost"


class MetadataTensorParameters(BaseModel):
    ext_datatype: Type


class MetadataTensor(BaseModel):
    name: str
    datatype: str
    shape: List[int]

    # NOTE: This is an addition to the standard
    parameters: Optional[MetadataTensorParameters] = None


class ModelDetails(BaseModel):
    name: str
    local_folder: str
    uri: str

    # TODO: Should we change this for `platform` to align with V2 dataplane?
    platform: ModelFramework

    inputs: Optional[List[MetadataTensor]] = []
    outputs: Optional[List[MetadataTensor]] = []
