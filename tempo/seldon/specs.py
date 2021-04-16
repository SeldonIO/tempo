import json

from tempo.kfserving.protocol import KFServingV1Protocol, KFServingV2Protocol
from tempo.seldon.constants import MLSERVER_IMAGE
from tempo.serve.constants import ENV_TEMPO_RUNTIME_OPTIONS
from tempo.serve.metadata import ModelDetails, ModelFramework, RuntimeOptions
from tempo.serve.runtime import ModelSpec

DefaultHTTPPort = "9000"
DefaultGRPCPort = "9500"

DefaultModelsPath = "/mnt/models"
DefaultServiceAccountName = "tempo-pipeline"


def get_container_spec(model_details: ModelSpec) -> dict:
    if model_details.model_details.platform == ModelFramework.TempoPipeline:
        return _V2ContainerFactory.get_container_spec(model_details.model_details, model_details.runtime_options)

    if isinstance(model_details.protocol, KFServingV2Protocol):
        return _V2ContainerFactory.get_container_spec(model_details.model_details, model_details.runtime_options)

    return _V1ContainerFactory.get_container_spec(model_details.model_details)


class _V1ContainerFactory:
    Images = {
        ModelFramework.SKLearn: "seldonio/sklearnserver:1.6.0-dev",
        ModelFramework.XGBoost: "seldonio/xgboostserver:1.6.0-dev",
        ModelFramework.Tensorflow: "tensorflow/serving:2.1.0",
    }

    @classmethod
    def get_container_spec(cls, model_details: ModelDetails) -> dict:
        model_image = cls.Images[model_details.platform]

        if model_details.platform == ModelFramework.Tensorflow:
            return {
                "image": model_image,
                "command": [
                    "--rest_api_port=" + DefaultHTTPPort,
                    "--model_name=" + model_details.name,
                    "--model_base_path=" + DefaultModelsPath,
                ],
            }
        else:
            parameters = [{"name": "model_uri", "value": DefaultModelsPath, "type": "STRING"}]
            env = {"PREDICTIVE_UNIT_PARAMETERS": json.dumps(parameters)}

            return {
                "image": model_image,
                "environment": env,
            }


class _V2ContainerFactory:
    MLServerImage = MLSERVER_IMAGE

    MLServerRuntimes = {
        ModelFramework.SKLearn: "mlserver_sklearn.SKLearnModel",
        ModelFramework.XGBoost: "mlserver_xgboost.XGBoostModel",
        ModelFramework.Custom: "tempo.mlserver.InferenceRuntime",
        ModelFramework.TempoPipeline: "tempo.mlserver.InferenceRuntime",
    }

    @classmethod
    def get_container_spec(cls, model_details: ModelDetails, runtime_options: RuntimeOptions) -> dict:
        mlserver_runtime = cls.MLServerRuntimes[model_details.platform]

        env = {
            "MLSERVER_HTTP_PORT": DefaultHTTPPort,
            "MLSERVER_GRPC_PORT": DefaultGRPCPort,
            "MLSERVER_MODEL_IMPLEMENTATION": mlserver_runtime,
            "MLSERVER_MODEL_NAME": model_details.name,
            "MLSERVER_MODEL_URI": DefaultModelsPath,
            ENV_TEMPO_RUNTIME_OPTIONS: json.dumps(runtime_options.dict()),
        }

        return {
            "image": cls.MLServerImage,
            "environment": env,
        }


class KubernetesSpec:
    Implementations = {
        ModelFramework.SKLearn: "SKLEARN_SERVER",
        ModelFramework.XGBoost: "XGBOOST_SERVER",
        ModelFramework.MLFlow: "MLFLOW_SERVER",
        ModelFramework.Tensorflow: "TENSORFLOW_SERVER",
        ModelFramework.PyTorch: "TRITON_SERVER",
        ModelFramework.ONNX: "TRITON_SERVER",
        ModelFramework.TensorRT: "TRITON_SERVER",
        # TODO: We need to set an implementation in order to get the init
        # container injected into the spec
        ModelFramework.TempoPipeline: "TRITON_SERVER",
    }

    def __init__(
        self,
        model_details: ModelSpec,
    ):
        self._details = model_details

    @property
    def spec(self) -> dict:
        predictor = self._get_predictor()
        protocol = self._get_spec_protocol()

        return {
            "apiVersion": "machinelearning.seldon.io/v1",
            "kind": "SeldonDeployment",
            "metadata": {
                "name": self._details.model_details.name,
                "namespace": self._details.runtime_options.k8s_options.namespace,
            },
            "spec": {"protocol": protocol, "predictors": [predictor]},
        }

    def _get_predictor(self) -> dict:
        # TODO: We need to insert `type: MODEL`, otherwise the validation
        # webhook complains
        graph = {
            "modelUri": self._details.model_details.uri,
            "name": self._details.model_details.name,
            "type": "MODEL",
        }

        if self._details.runtime_options.k8s_options.authSecretName:
            graph["envSecretRefName"] = self._details.runtime_options.k8s_options.authSecretName

        if self._details.model_details.platform in self.Implementations:
            model_implementation = self.Implementations[self._details.model_details.platform]
            graph["implementation"] = model_implementation

        if self._details.model_details.platform == ModelFramework.TempoPipeline:
            serviceAccountName = self._details.runtime_options.k8s_options.serviceAccountName
            if serviceAccountName is None:
                serviceAccountName = DefaultServiceAccountName
            graph["serviceAccountName"] = serviceAccountName

        predictor = {
            "graph": graph,
            "name": "default",
            "replicas": self._details.runtime_options.k8s_options.replicas,
        }

        if self._details.model_details.platform == ModelFramework.TempoPipeline:
            predictor["componentSpecs"] = self._get_component_specs()

        return predictor

    def _get_component_specs(self) -> list:
        container_spec = get_container_spec(self._details)
        container_env = [{"name": name, "value": value} for name, value in container_spec["environment"].items()]

        return [
            {
                "spec": {
                    "containers": [
                        {
                            "name": self._details.model_details.name,
                            "image": container_spec["image"],
                            "env": container_env,
                            # TODO: Necessary to override Triton defaults (see
                            # note above)
                            "args": [],
                        }
                    ]
                }
            }
        ]

    def _get_spec_protocol(self) -> str:
        if isinstance(self._details.protocol, KFServingV2Protocol):
            return "kfserving"

        if isinstance(self._details.protocol, KFServingV1Protocol):
            return "tensorflow"

        return "seldon"
