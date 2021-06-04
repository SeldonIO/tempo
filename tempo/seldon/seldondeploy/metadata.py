import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from google.protobuf.json_format import ParseDict
from protoc_gen_validate.validator import validate
from seldon_deploy_sdk import ApiClient, ModelMetadataServiceApi, V1Model, V1PredictionSchema
from seldon_deploy_sdk.models.v1_artifact_type import V1ArtifactType

from tempo.metadata.prediction_schema_pb2 import PredictionSchema
from tempo.serve.base import BaseModel
from tempo.serve.metadata import ModelFramework


class Metadata:
    def __init__(self, api_client: ApiClient):
        self.api_client = api_client

    def _get_artifact_type(self, model_framework: ModelFramework) -> V1ArtifactType:
        if model_framework == ModelFramework.SKLearn:
            return V1ArtifactType.SKLEARN
        elif model_framework == ModelFramework.XGBoost:
            return V1ArtifactType.XGBOOST
        elif model_framework == ModelFramework.ONNX:
            return V1ArtifactType.ONNX
        elif model_framework == ModelFramework.Tensorflow:
            return V1ArtifactType.TENSORFLOW
        elif model_framework == ModelFramework.TensorRT:
            return V1ArtifactType.TENSORRT
        elif model_framework == ModelFramework.PyTorch:
            return V1ArtifactType.PYTORCH
        elif model_framework == ModelFramework.MLFlow:
            return V1ArtifactType.MLFLOW
        else:
            return V1ArtifactType.CUSTOM

    def _load_metadata(self, folder: str) -> V1PredictionSchema:
        metadata_file = os.path.join(folder, "metadata.json")
        path = Path(metadata_file)
        if path.is_file():
            with open(metadata_file, "r") as f:
                raw = f.read()
                j = json.loads(raw)
                ps = PredictionSchema()
                ParseDict(j, ps)
                validate(ps)
                # construct V1PredictionSchema. Need to use hack to create Fake Requests response
                res = SimpleNamespace(data=raw)
                return self.api_client.deserialize(res, V1PredictionSchema)
        else:
            return None

    def register(self, model: Any):
        spec: BaseModel = model.get_tempo()
        prediction_schema = self._load_metadata(spec.details.local_folder)
        meta = V1Model(
            uri=spec.details.uri,
            name=spec.details.name,
            version=spec.details.version,
            artifact_type=self._get_artifact_type(spec.details.platform),
            task_type=spec.details.task_type,
            tags={
                "description": spec.details.description,
            },
            prediction_schema=prediction_schema,
        )
        api_instance = ModelMetadataServiceApi(self.api_client)
        # Create a Model Metadata entry.
        api_response = api_instance.model_metadata_service_create_model_metadata(meta)
        print(api_response)
