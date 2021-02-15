from pydantic import BaseModel
from enum import Enum
from typing import Optional, List, Type, Dict, Union
from tempo.serve.constants import ModelDataType


class ModelFramework(Enum):
    SKLearn = "sklearn"
    XGBoost = "xgboost"
    MLFlow = "mlflow"
    Tensorflow = "tensorflow"
    PyTorch = "pytorch"
    ONNX = "ONNX"
    TensorRT = "tensorrt"
    Custom = "custom"
    TempoPipeline = "tempo"


class ModelDataArg(BaseModel):

    ty: Type
    name: str = None


class ModelDataArgs(BaseModel):

    args: List[ModelDataArg]

    def _get_type_by_name(self, name: str) -> Optional[Type]:
        for arg in self.args:
            if name == arg.name:
                return arg.ty
        return None

    def __getitem__(self, idx: Union[str, int]) -> Optional[Type]:
        if type(idx) == str:
            return self._get_type_by_name(idx)
        else:
            if idx < len(self.args):
                return self.args[idx].ty
            else:
                return None

    def __len__(self):
        return len(self.args)


class ModelDetails(BaseModel):
    name: str
    local_folder: str
    uri: str
    platform: ModelFramework
    inputs: ModelDataArgs
    outputs: ModelDataArgs


class KubernetesOptions(BaseModel):
    replicas: int = 1
    namespace = "default"
    minReplicas: int = None
    maxReplicas: int = None
