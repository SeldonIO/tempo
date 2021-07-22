import json
import os
from inspect import iscoroutinefunction

from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.utils import get_model_uri

from tempo.magic import PayloadContext, TempoContextWrapper, tempo_context

from .insights.manager import InsightsManager
from .insights.wrapper import InsightsWrapper
from .serve.base import BaseModel
from .serve.constants import ENV_TEMPO_RUNTIME_OPTIONS
from .serve.loader import load
from .serve.metadata import InsightRequestModes, InsightsTypes, ModelFramework, dict_to_runtime
from .serve.utils import PredictMethodAttr
from .state.state import BaseState


def _needs_init(model: BaseModel):
    is_class = model._K is not None
    has_annotation = hasattr(model._user_func, PredictMethodAttr)
    is_bound = hasattr(model._user_func, "__self__")

    return is_class and has_annotation and not is_bound


class InferenceRuntime(MLModel):
    async def load(self) -> bool:
        self._model = await self._load_model()
        await self._load_runtime()
        await self._load_insights()
        await self._load_state()

        self._is_coroutine = iscoroutinefunction(self._model.request)

        self.ready = True
        return self.ready

    async def _load_model(self) -> BaseModel:
        model_uri = await get_model_uri(self._settings)

        model = load(model_uri)
        model.details.local_folder = model_uri

        if model.details.platform == ModelFramework.TempoPipeline:
            # If pipeline, call children models remotely
            model.set_remote(True)

        if _needs_init(model):
            instance = model._K()
            # Make sure that the model is the instance's model (and not the
            # class attribute)
            model = instance.get_tempo()

        if model._load_func:
            model._load_func()

        return model

    async def _load_state(self):
        runtime_options = self._model.runtime_options_override
        if not runtime_options:
            runtime_options = self._model.model_spec.runtime_options

        self.state = BaseState.from_conf(runtime_options.state_options)

    async def _load_insights(self):
        runtime_options = self._model.runtime_options_override
        if not runtime_options:
            runtime_options = self._model.model_spec.runtime_options
        insights_params = runtime_options.insights_options.dict()

        self.insights_manager = InsightsManager(**insights_params)

    async def _load_runtime(self):
        rt_options_str = os.getenv(ENV_TEMPO_RUNTIME_OPTIONS)
        if rt_options_str:
            rt_options = dict_to_runtime(json.loads(rt_options_str))
            self._model.set_runtime_options_override(rt_options)

    async def predict(self, request: InferenceRequest) -> InferenceResponse:

        request_dict = request.dict()

        insights_wrapper = InsightsWrapper(self.insights_manager)
        # TODO: Add request_id, response_headers, request_headers, etc
        payload_context = PayloadContext(request_id=request.id, request=request_dict)
        tempo_wrapper = TempoContextWrapper(payload_context, insights_wrapper, self.state)
        tempo_context.set(tempo_wrapper)

        response_dict = self._model.request(request_dict)
        if self._is_coroutine:
            response_dict = await response_dict  # type: ignore

        # TODO: Ensure model_version is added by mlserver
        response_dict["model_version"] = "NOTIMPLEMENTED"

        # TODO: Move to functions declared upfront with logic contained to avoid if
        if self._model.get_insights_mode == InsightRequestModes.ALL:
            insights_wrapper.log(request_dict, insights_type=InsightsTypes.INFER_REQUEST)
            insights_wrapper.log(response_dict, insights_type=InsightsTypes.INFER_RESPONSE)
        else:
            if self._model.get_insights_mode == InsightRequestModes.REQUEST or insights_wrapper.set_log_request:
                insights_wrapper.log(request_dict, insights_type=InsightsTypes.INFER_REQUEST)
            if self._model.get_insights_mode == InsightRequestModes.RESPONSE or insights_wrapper.set_log_response:
                insights_wrapper.log(response_dict, insights_type=InsightsTypes.INFER_RESPONSE)

        return InferenceResponse(**response_dict)
