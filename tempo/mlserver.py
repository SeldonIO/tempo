from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.utils import get_model_uri

from tempo.serve.loader import load


class InferenceRuntime(MLModel):
    async def load(self) -> bool:
        pipeline_uri = await get_model_uri(self._settings)
        self._pipeline = load(pipeline_uri)
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        prediction = self._pipeline.request(payload.dict())
        return InferenceResponse(**prediction)
