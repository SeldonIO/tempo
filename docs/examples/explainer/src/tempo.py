import os
import dill
import numpy as np
from alibi.utils.wrappers import ArgmaxTransformer
from src.constants import ARTIFACTS_FOLDER, EXPLAINER_FOLDER, MODEL_FOLDER

from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.serve.pipeline import PipelineModels
from tempo.serve.utils import pipeline, predictmethod


def create_adult_model() -> Model :
    sklearn_model = Model(
        name="income-sklearn",
        platform=ModelFramework.SKLearn,
        local_folder=os.path.join(ARTIFACTS_FOLDER, MODEL_FOLDER),
        uri="gs://seldon-models/test/income/model",
    )

    return sklearn_model


def create_explainer(model: Model):

    @pipeline(
        name="income-explainer",
        uri="s3://tempo/explainer/pipeline",
        local_folder=os.path.join(ARTIFACTS_FOLDER, EXPLAINER_FOLDER),
        models=PipelineModels(sklearn=model),
    )
    class ExplainerPipeline(object):
        def __init__(self):
            pipeline = self.get_tempo()
            models_folder = pipeline.details.local_folder

            explainer_path = os.path.join(models_folder, "explainer.dill")
            with open(explainer_path, "rb") as f:
                self.explainer = dill.load(f)

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
            self.update_predict_fn(payload)
            explanation = self.explainer.explain(payload, **parameters)
            return explanation.to_json()

    #explainer = ExplainerPipeline()
    #return sklearn_model, explainer
    return ExplainerPipeline
