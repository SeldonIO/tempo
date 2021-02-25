import json

from tempo.serve.metadata import ModelDetails, ModelFramework
from tempo.serve.protocol import Protocol
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.metadata import KubernetesOptions

DefaultHTTPPort = "9000"
DefaultGRPCPort = "9500"

DefaultModelsPath = "/mnt/models"


def get_container_spec(model_details: ModelDetails, protocol: Protocol) -> dict:
    if model_details.platform == ModelFramework.TempoPipeline:
        return _V2ContainerFactory.get_container_spec(model_details)

    if isinstance(protocol, KFServingV2Protocol):
        return _V2ContainerFactory.get_container_spec(model_details)

    return _V1ContainerFactory.get_container_spec(model_details)


class _V1ContainerFactory:
    Images = {
        ModelFramework.SKLearn: "seldonio/sklearnserver:1.6.0-dev",
        ModelFramework.XGBoost: "seldonio/xgboostserver:1.6.0-dev",
    }

    @classmethod
    def get_container_spec(cls, model_details: ModelDetails) -> dict:
        model_image = cls.Images[model_details.platform]

        parameters = [
            {"name": "model_uri", "value": DefaultModelsPath, "type": "STRING"}
        ]
        env = {"PREDICTIVE_UNIT_PARAMETERS": json.dumps(parameters)}

        return {
            "image": model_image,
            "environment": env,
        }


class _V2ContainerFactory:
    MLServerImage = "seldonio/mlserver:0.3.1.dev4"

    MLServerRuntimes = {
        ModelFramework.SKLearn: "mlserver_sklearn.SKLearnModel",
        ModelFramework.XGBoost: "mlserver_xgboost.XGBoostModel",
        ModelFramework.TempoPipeline: "mlserver_tempo.TempoModel",
    }

    @classmethod
    def get_container_spec(cls, model_details: ModelDetails) -> dict:
        mlserver_runtime = cls.MLServerRuntimes[model_details.platform]

        env = {
            "MLSERVER_HTTP_PORT": DefaultHTTPPort,
            "MLSERVER_GRPC_PORT": DefaultGRPCPort,
            "MLSERVER_MODEL_IMPLEMENTATION": mlserver_runtime,
            "MLSERVER_MODEL_NAME": model_details.name,
            "MLSERVER_MODEL_URI": DefaultModelsPath,
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
        model_details: ModelDetails,
        protocol: Protocol,
        k8s_options: KubernetesOptions,
    ):
        self._details = model_details
        self._protocol = protocol
        self._k8s_options = k8s_options

    @property
    def spec(self) -> dict:
        predictor = self._get_predictor()
        protocol = self._get_spec_protocol()

        return {
            "apiVersion": "machinelearning.seldon.io/v1",
            "kind": "SeldonDeployment",
            "metadata": {
                "name": self._details.name,
                "namespace": self._k8s_options.namespace,
            },
            "spec": {"protocol": protocol, "predictors": [predictor]},
        }

    def _get_predictor(self) -> dict:
        # TODO: We need to insert `type: MODEL`, otherwise the validation
        # webhook complains
        graph = {"modelUri": self._details.uri, "name": "classifier", "type": "MODEL"}

        if self._details.platform in self.Implementations:
            model_implementation = self.Implementations[self._details.platform]
            graph["implementation"] = model_implementation

        predictor = {
            "graph": graph,
            "name": "default",
            "replicas": self._k8s_options.replicas,
        }

        if self._details.platform == ModelFramework.TempoPipeline:
            predictor["componentSpecs"] = self._get_component_specs()

        return predictor

    def _get_component_specs(self) -> list:
        container_spec = get_container_spec(self._details, self._protocol)
        container_env = [
            {"name": name, "value": value}
            for name, value in container_spec["environment"].items()
        ]

        return [
            {
                "spec": {
                    "containers": [
                        {
                            "name": "classifier",
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
        if isinstance(self._protocol, KFServingV2Protocol):
            return "kfserving"

        return "seldon"
