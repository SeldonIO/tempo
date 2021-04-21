import json
import os

import numpy as np
from alibi_detect.base import NumpyEncoder
from src.constants import MODEL_FOLDER, OUTLIER_FOLDER

from tempo.kfserving.protocol import KFServingV1Protocol, KFServingV2Protocol
from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.serve.pipeline import PipelineModels
from tempo.serve.utils import model, pipeline, predictmethod


def create_outlier_cls(artifacts_folder: str):
    @model(
        name="outlier",
        platform=ModelFramework.TempoPipeline,
        protocol=KFServingV2Protocol(),
        uri="s3://tempo/outlier/cifar10/outlier",
        local_folder=f"{artifacts_folder}/{OUTLIER_FOLDER}",
    )
    class OutlierModel(object):
        def __init__(self):
            self.loaded = False

        def load(self):
            from alibi_detect.utils.saving import load_detector

            if "MLSERVER_MODELS_DIR" in os.environ:
                models_folder = "/mnt/models"
            else:
                models_folder = f"{artifacts_folder}/{OUTLIER_FOLDER}"
            print(f"Loading from {models_folder}")
            self.od = load_detector(f"{models_folder}/cifar10")
            self.loaded = True

        def unload(self):
            self.od = None
            self.loaded = False

        @predictmethod
        def outlier(self, payload: np.ndarray) -> dict:
            if not self.loaded:
                self.load()
            od_preds = self.od.predict(
                payload,
                outlier_type="instance",  # use 'feature' or 'instance' level
                return_feature_score=True,
                # scores used to determine outliers
                return_instance_score=True,
            )

            return json.loads(json.dumps(od_preds, cls=NumpyEncoder))

    return OutlierModel


def create_model(arifacts_folder: str):

    cifar10_model = Model(
        name="resnet32",
        protocol=KFServingV1Protocol(),
        platform=ModelFramework.Tensorflow,
        uri="gs://seldon-models/tfserving/cifar10/resnet32",
        local_folder=f"{arifacts_folder}/{MODEL_FOLDER}",
    )

    return cifar10_model


def create_svc_cls(outlier, model, arifacts_folder: str):
    @pipeline(
        name="cifar10-service",
        protocol=KFServingV2Protocol(),
        uri="s3://tempo/outlier/cifar10/svc",
        local_folder=f"{arifacts_folder}/svc",
        models=PipelineModels(outlier=outlier, cifar10=model),
    )
    class Cifar10Svc(object):
        @predictmethod
        def predict(self, payload: np.ndarray) -> np.ndarray:
            r = self.models.outlier(payload=payload)
            if r["data"]["is_outlier"][0]:
                return np.array([])
            else:
                return self.models.cifar10(payload)

    return Cifar10Svc
