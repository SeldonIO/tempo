import abc
from enum import Enum
from pydoc import locate
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, validator


class InsightsTypes(Enum):
    INFER_REQUEST: str = "io.seldon.serving.inference.request"
    INFER_RESPONSE: str = "io.seldon.serving.inference.response"
    CUSTOM_INSIGHT: str = "io.seldon.serving.inference.custominsight"


DEFAULT_INSIGHTS_TYPE = InsightsTypes.CUSTOM_INSIGHT


def dict_to_runtime(d: Dict) -> "BaseRuntimeOptionsType":
    runtime = d.get("runtime")
    # TODO update to load class by adding type to each
    enterprise_runtimes = ["tempo.seldon.SeldonDeployRunime"]
    docker_runtimes = ["tempo.seldon.SeldonDockerRuntime"]
    k8s_runtimes = ["tempo.kfserving.KFServingKubernetesRuntime", "tempo.seldon.SeldonKubernetesRuntime"]
    if runtime in k8s_runtimes:
        return KubernetesRuntimeOptions(**d)
    elif runtime in enterprise_runtimes:
        return EnterpriseRuntimeOptions(**d)
    elif runtime in docker_runtimes:
        return DockerOptions(**d)
    else:
        raise Exception("Runtime not supported: " + str(runtime))


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

    @validator("ty", pre=True)
    def ensure_type(cls, v):
        if isinstance(v, str):
            return locate(v)
        else:
            return v

    class Config:
        json_encoders = {type: lambda v: v.__module__ + "." + v.__name__}


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

    class Config:
        json_encoders = {type: lambda v: v.__module__ + "." + v.__name__}


class ClientDetails(BaseModel):

    url: str
    headers: Dict[str, str]
    verify_ssl: bool


class ModelDetails(BaseModel):
    name: str
    local_folder: str
    uri: str
    platform: ModelFramework
    inputs: ModelDataArgs
    outputs: ModelDataArgs
    description: str = ""

    # class Config:
    #    use_enum_values = True


class InsightRequestModes(Enum):
    ALL = "ALL"
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    NONE = "NONE"


DEFAULT_INSIGHTS_REQUEST_MODES = InsightRequestModes.NONE


class InsightsOptions(BaseModel):
    worker_endpoint: str = ""
    batch_size: int = 1
    parallelism: int = 1
    retries: int = 3
    window_time: int = 0
    mode_type: InsightRequestModes = DEFAULT_INSIGHTS_REQUEST_MODES
    in_asyncio: bool = False

    class Config:
        # Required to ensure enum json serialisation https://pydantic-docs.helpmanual.io/usage/model_config/
        use_enum_values = True


class InsightsPayload(BaseModel):
    request_id: str = ""
    data: Any = None
    insights_type: InsightsTypes = DEFAULT_INSIGHTS_TYPE

    class Config:
        use_enum_values = True


class StateTypes(Enum):
    LOCAL = "LOCAL"
    REDIS = "REDIS"


class StateOptions(BaseModel):
    state_type: Optional[StateTypes] = StateTypes.LOCAL
    key_prefix: str = ""
    host: str = ""
    port: str = ""

    class Config:
        use_enum_values = True


class IngressOptions(BaseModel):
    ingress: str = "tempo.ingress.istio.IstioIngress"
    ssl: bool = False
    verify_ssl: bool = True


class _BaseRuntimeOptions(BaseModel):
    runtime: str = ""
    state_options: StateOptions = StateOptions()
    insights_options: InsightsOptions = InsightsOptions()
    ingress_options: IngressOptions = IngressOptions()

    class Config:
        use_enum_values = True


class EnterpriseRuntimeAuthType(Enum):
    session_cookie = "session_cookie"
    oidc = "oidc"


class EnterpriseRuntimeOptions(_BaseRuntimeOptions):
    runtime: str = "tempo.seldon.SeldonDeployRunime"
    host: str
    user: str
    password: str
    auth_type: EnterpriseRuntimeAuthType = EnterpriseRuntimeAuthType.session_cookie
    oidc_client_id: str
    oidc_server: str
    verify_ssl: bool = True


class KubernetesRuntimeOptions(_BaseRuntimeOptions):
    runtime: str = "tempo.seldon.SeldonKubernetesRuntime"
    # Kubernetes parameters
    replicas: int = 1
    namespace = "default"
    minReplicas: Optional[int] = None
    maxReplicas: Optional[int] = None
    authSecretName: Optional[str] = None
    serviceAccountName: Optional[str] = None
    # TODO move to separate seldonkubernetesruntime
    add_svc_orchestrator: bool = False


class DockerOptions(_BaseRuntimeOptions):
    runtime: str = "tempo.seldon.SeldonDockerRuntime"


class _BaseProductOptions(BaseModel, abc.ABC):
    # ty: str = ""
    local_options: DockerOptions = DockerOptions()
    remote_options: Any = KubernetesRuntimeOptions()


class SeldonCoreOptions(_BaseProductOptions):
    # ty: str = "tempo.serve.metadata.SeldonCoreOptions"
    remote_options: KubernetesRuntimeOptions = KubernetesRuntimeOptions(runtime="tempo.seldon.SeldonKubernetesRuntime")


class KFServingOptions(_BaseProductOptions):
    # ty: str = "tempo.serve.metadata.KFServingOptions"
    remote_options: KubernetesRuntimeOptions = KubernetesRuntimeOptions(
        runtime="tempo.kfserving.KFservingKubernetesRuntime"
    )


class SeldonEnterpriseOptions(_BaseProductOptions):
    # ty: str = "tempo.serve.metadata.KFServingOptions"
    remote_options: EnterpriseRuntimeOptions


BaseRuntimeOptionsType = Union[KubernetesRuntimeOptions, DockerOptions, EnterpriseRuntimeOptions]
BaseProductOptionsType = Union[SeldonCoreOptions, KFServingOptions, SeldonEnterpriseOptions]
