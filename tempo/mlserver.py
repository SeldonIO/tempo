import json
import os

from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.utils import get_model_uri

from tempo.serve.constants import ENV_TEMPO_RUNTIME_OPTIONS
from tempo.serve.loader import load
from tempo.serve.metadata import RuntimeOptions


class InferenceRuntime(MLModel):
    async def load(self) -> bool:
        pipeline_uri = await get_model_uri(self._settings)
        self._pipeline = load(pipeline_uri)
        rt_options_str = os.getenv(ENV_TEMPO_RUNTIME_OPTIONS)
        if rt_options_str:
            rt_options = RuntimeOptions(**json.loads(rt_options_str))
            self._pipeline.set_runtime_options_override(rt_options)
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        prediction = self._pipeline.request(payload.dict())
        return InferenceResponse(**prediction)
