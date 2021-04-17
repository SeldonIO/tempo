import os
from typing import Any, Tuple

import dill
import numpy as np
from alibi.utils.wrappers import ArgmaxTransformer
from src.constants import EXPLAINER_FOLDER, MODEL_FOLDER

from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.serve.pipeline import PipelineModels
from tempo.serve.utils import pipeline, predictmethod


def create_tempo_artifacts(artifacts_folder: str) -> Tuple[Model, Any]:
    sklearn_model = Model(
        name="income-sklearn",
        platform=ModelFramework.SKLearn,
        local_folder=f"{artifacts_folder}/{MODEL_FOLDER}",
        uri="gs://seldon-models/test/income/model",
    )

    @pipeline(
        name="income-explainer",
        uri="s3://tempo/explainer/pipeline",
        local_folder=f"{artifacts_folder}/{EXPLAINER_FOLDER}",
        models=PipelineModels(sklearn=sklearn_model),
    )
    class ExplainerPipeline(object):
        def __init__(self):
            if "MLSERVER_MODELS_DIR" in os.environ:
                models_folder = ""
            else:
                models_folder = f"{artifacts_folder}/{EXPLAINER_FOLDER}"
            with open(models_folder + "/explainer.dill", "rb") as f:
                self.explainer = dill.load(f)
            self.ran_init = True

        def update_predict_fn(self, x):
            if np.argmax(self.models.sklearn(x).shape) == 0:
                self.explainer.predictor = self.models.sklearn
                self.explainer.samplers[0].predictor = self.models.sklearn
            else:
                self.explainer.predictor = ArgmaxTransformer(self.models.sklearn)
                self.explainer.samplers[0].predictor = ArgmaxTransformer(self.models.sklearn)

        @predictmethod
        def explain(self, payload: np.ndarray, parameters: dict) -> str:
            print("Explain called with ", parameters)
            if not self.ran_init:
                print("Loading explainer")
                self.__init__()
            self.update_predict_fn(payload)
            explanation = self.explainer.explain(payload, **parameters)
            return explanation.to_json()

    explainer = ExplainerPipeline()
    return sklearn_model, explainer
