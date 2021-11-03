import json

from tempo.k8s.constants import TempoK8sDescriptionAnnotation, TempoK8sLabel, TempoK8sModelSpecAnnotation
from tempo.protocols.tensorflow import TensorflowProtocol
from tempo.protocols.v2 import V2Protocol
from tempo.seldon.constants import MLSERVER_IMAGE, TRITON_IMAGE
from tempo.serve.base import ModelSpec
from tempo.serve.constants import ENV_TEMPO_RUNTIME_OPTIONS
from tempo.serve.metadata import BaseRuntimeOptionsType, KubernetesRuntimeOptions, ModelDetails, ModelFramework

DefaultHTTPPort = "9000"
DefaultGRPCPort = "9500"

DefaultModelsPath = "/mnt/models"
DefaultServiceAccountName = "tempo-pipeline"


def get_container_spec(model_details: ModelSpec) -> dict:
    runtime_options = model_details.runtime_options.copy(deep=True)
    # Ensure running inside asyncio loop in MLServer
    runtime_options.insights_options.in_asyncio = True
    if (
        model_details.model_details.platform == ModelFramework.TempoPipeline
        or model_details.model_details.platform == ModelFramework.Custom
    ):
        return _V2ContainerFactory.get_container_spec(model_details.model_details, runtime_options)

    if isinstance(model_details.protocol, V2Protocol):
        return _V2ContainerFactory.get_container_spec(model_details.model_details, runtime_options)

    return _V1ContainerFactory.get_container_spec(model_details.model_details)


class _V1ContainerFactory:
    Images = {
        ModelFramework.SKLearn: "seldonio/sklearnserver:1.6.0-dev",
        ModelFramework.XGBoost: "seldonio/xgboostserver:1.6.0-dev",
        ModelFramework.Tensorflow: "tensorflow/serving:2.1.0",
        ModelFramework.TensorRT: "nvcr.io/nvidia/tritonserver:21.08-py3",
        ModelFramework.ONNX: "nvcr.io/nvidia/tritonserver:21.08-py3",
        ModelFramework.PyTorch: "nvcr.io/nvidia/tritonserver:21.08-py3",
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
    TritonImage = TRITON_IMAGE

    MLServerRuntimes = {
        ModelFramework.MLFlow: "mlserver_mlflow.MLflowRuntime",
        ModelFramework.SKLearn: "mlserver_sklearn.SKLearnModel",
        ModelFramework.XGBoost: "mlserver_xgboost.XGBoostModel",
        ModelFramework.Custom: "tempo.mlserver.InferenceRuntime",
        ModelFramework.TempoPipeline: "tempo.mlserver.InferenceRuntime",
    }

    @classmethod
    def get_container_spec(cls, model_details: ModelDetails, runtime_options: BaseRuntimeOptionsType) -> dict:
        if (
            model_details.platform == ModelFramework.PyTorch
            or model_details.platform == ModelFramework.TensorRT
            or model_details.platform == ModelFramework.ONNX
            or model_details.platform == ModelFramework.Tensorflow
        ):
            return {
                "image": cls.TritonImage,
                "command": [
                    "/opt/tritonserver/bin/tritonserver",
                    f"--grpc-port={DefaultGRPCPort}",
                    f"--http-port={DefaultHTTPPort}",
                    f"--model-repository={DefaultModelsPath}",
                    "--strict-model-config=false",
                ],
            }
        else:
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
        ModelFramework.TempoPipeline: "TEMPO_SERVER",
        ModelFramework.Custom: "TEMPO_SERVER",
    }

    def __init__(
        self,
        model_details: ModelSpec,
        runtime_options: KubernetesRuntimeOptions,
    ):
        self._details = model_details
        self._runtime_options = runtime_options

    @property
    def spec(self) -> dict:
        predictor = self._get_predictor()
        protocol = self._get_spec_protocol()

        return {
            "apiVersion": "machinelearning.seldon.io/v1",
            "kind": "SeldonDeployment",
            "metadata": {
                "name": self._details.model_details.name,
                "namespace": self._details.runtime_options.namespace,  # type: ignore
                "labels": {
                    TempoK8sLabel: "true",
                },
                "annotations": {
                    TempoK8sDescriptionAnnotation: self._details.model_details.description,
                    TempoK8sModelSpecAnnotation: self._details.json(),
                },
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

        if self._details.runtime_options.authSecretName:  # type: ignore
            graph["envSecretRefName"] = self._details.runtime_options.authSecretName  # type: ignore

        if self._details.model_details.platform in self.Implementations:
            model_implementation = self.Implementations[self._details.model_details.platform]
            graph["implementation"] = model_implementation

        if (
            self._details.model_details.platform == ModelFramework.TempoPipeline
            or self._details.model_details.platform == ModelFramework.Custom
        ):
            serviceAccountName = self._details.runtime_options.serviceAccountName  # type: ignore
            if serviceAccountName is None:
                serviceAccountName = DefaultServiceAccountName
            graph["serviceAccountName"] = serviceAccountName

        predictor = {
            "graph": graph,
            "name": "default",
            "replicas": self._details.runtime_options.replicas,  # type: ignore
        }

        if not self._runtime_options.add_svc_orchestrator:  # type: ignore
            predictor["annotations"] = {
                "seldon.io/no-engine": "true",
            }

        if (
            self._details.model_details.platform == ModelFramework.TempoPipeline
            or self._details.model_details.platform == ModelFramework.Custom
        ):
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
        if isinstance(self._details.protocol, V2Protocol):
            return "kfserving"

        if isinstance(self._details.protocol, TensorflowProtocol):
            return "tensorflow"

        return "seldon"
