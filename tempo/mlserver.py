import json
import os

from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.utils import get_model_uri

from .serve.base import BaseModel
from .serve.constants import ENV_TEMPO_RUNTIME_OPTIONS
from .serve.loader import load
from .serve.metadata import ModelFramework, RuntimeOptions
from .serve.utils import PredictMethodAttr


def _is_class(model: BaseModel) -> bool:
    return hasattr(model._user_func, PredictMethodAttr)


class InferenceRuntime(MLModel):
    async def load(self) -> bool:
        self._model = await self._load_model()
        await self._load_runtime()

        self.ready = True
        return self.ready

    async def _load_model(self) -> BaseModel:
        model_uri = await get_model_uri(self._settings)

        model = load(model_uri)
        model.details.local_folder = model_uri

        if model.details.platform == ModelFramework.TempoPipeline:
            # If pipeline, call children models remotely
            model.set_remote(True)

        if _is_class(model):
            # TODO: Call __init__()
            pass

        if model._load_func:
            model._load_func()

        return model

    async def _load_runtime(self):
        rt_options_str = os.getenv(ENV_TEMPO_RUNTIME_OPTIONS)
        if rt_options_str:
            rt_options = RuntimeOptions(**json.loads(rt_options_str))
            self._model.set_runtime_options_override(rt_options)

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        prediction = self._model.request(payload.dict())
        return InferenceResponse(**prediction)
