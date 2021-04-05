from enum import Enum
from typing import List, Optional, Type, Union

from pydantic import BaseModel


class ModelFramework(Enum):
    SKLearn = "sklearn"
    XGBoost = "xgboost"
    MLFlow = "mlflow"
    Tensorflow = "tensorflow"
    PyTorch = "pytorch"
    ONNX = "ONNX"
    TensorRT = "tensorrt"
    Alibi = "alibi"
    Custom = "custom"
    TempoPipeline = "tempo"


class ModelDataArg(BaseModel):
    ty: Type
    name: Optional[str] = None


class ModelDataArgs(BaseModel):

    args: List[ModelDataArg]

    def _get_type_by_name(self, name: str) -> Optional[Type]:
        for arg in self.args:
            if name == arg.name:
                return arg.ty
        return None

    def __getitem__(self, idx: Union[str, int]) -> Optional[Type]:
        if isinstance(idx, str):
            return self._get_type_by_name(idx)

        if idx < len(self.args):
            # NOTE: `idx` here must be an int
            return self.args[idx].ty

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
    minReplicas: Optional[int] = None
    maxReplicas: Optional[int] = None
    authSecretName: Optional[str] = None
    serviceAccountName: Optional[str] = None
    defaultRuntime: str = "tempo.seldon.SeldonKubernetesRuntime"


class DockerOptions(BaseModel):
    defaultRuntime: str = "tempo.seldon.SeldonDockerRuntime"


class IngressOptions(BaseModel):
    ingress: str = "tempo.ingress.istio.IstioIngress"
    ssl: bool = False
    verify_ssl: bool = True


class RuntimeOptions(BaseModel):
    runtime: Optional[str] = None
    docker_options: DockerOptions = DockerOptions()
    k8s_options: KubernetesOptions = KubernetesOptions()
    ingress_options: IngressOptions = IngressOptions()
