import json

from tempo.serve.metadata import ModelDetails, ModelFramework
from tempo.serve.protocol import Protocol
from tempo.kfserving.protocol import KFServingV2Protocol

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
    MLServerImage = "seldonio/mlserver:0.3.1.dev2"

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
