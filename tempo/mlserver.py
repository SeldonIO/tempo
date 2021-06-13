import json
import os
from inspect import iscoroutinefunction
import contextvars

from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.utils import get_model_uri

from .serve.base import BaseModel
from .serve.constants import ENV_TEMPO_RUNTIME_OPTIONS
from .serve.loader import load
from .serve.metadata import ModelFramework, RuntimeOptions
from .serve.utils import PredictMethodAttr
from .utils import logger

from .insights.manager import InsightsManager


def _needs_init(model: BaseModel):
    is_class = model._K is not None
    has_annotation = hasattr(model._user_func, PredictMethodAttr)
    # TODO: find out why this is required
    # is_bound = hasattr(model._user_func, "__self__")

    logger.warning(f"isclass {is_class} hasannot {has_annotation}")
    return is_class and has_annotation# and not is_bound


insights_context = contextvars.ContextVar("insights_manager", default=None)

class InferenceRuntime(MLModel):
    async def load(self) -> bool:
        self._model = await self._load_model()
        await self._load_runtime()
        await self._load_insights()

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

            runtime_options = model.runtime_options_override
            if not runtime_options:
                runtime_options = model.model_spec.runtime_options
            insights_params = runtime_options.insights_options.dict()

            insights_manager = InsightsManager(**insights_params)
            model._K.pipeline.insights_manager = insights_manager
            model._K.pipeline.insights = insights_manager
            model._K.pipeline.insights_context = insights_context

            instance = model._K()
            # Make sure that the model is the instance's model (and not the
            # class attribute)
            model = instance.get_tempo()

        if model._load_func:
            model._load_func()

        return model

    async def _load_insights(self):
        runtime_options = self._model.runtime_options_override
        if not runtime_options:
            runtime_options = self._model.model_spec.runtime_options
        insights_params = runtime_options.insights_options.dict()

        self._model.insights_manager = InsightsManager(**insights_params)

    async def _load_runtime(self):
        rt_options_str = os.getenv(ENV_TEMPO_RUNTIME_OPTIONS)
        if rt_options_str:
            rt_options = RuntimeOptions(**json.loads(rt_options_str))
            self._model.set_runtime_options_override(rt_options)

<<<<<<< HEAD
    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        prediction = self._model.request(payload.dict())
        if self._is_coroutine:
            prediction = await prediction  # type: ignore

        return InferenceResponse(**prediction)
=======
        self._model.set_insights_context(insights_context)


    async def predict(self, request: InferenceRequest) -> InferenceResponse:

        class InsightsWrapper:
            def __init__(self, manager):
                self.set_log_request = False
                self.set_log_response = False
                self._manager = manager
            def log(self, data):
                self._manager.log(data)
            def log_request(self):
                self.set_log_request = True
            def log_response(self):
                self.set_log_response = True

        insights_wrapper = InsightsWrapper(self._model.insights_manager)
        logger.warning("setting manager context")
        logger.warning(f"value of model {self._model.insights_context}")
        insights_context.set(insights_wrapper)

        prediction = self._model.request(request.dict())
        response = InferenceResponse(**prediction)

        if insights_wrapper:
            if insights_wrapper.set_log_request:
                self._model.insights_manager.log(request.dict())
            if insights_wrapper.set_log_response:
                self._model.insights_manager.log(response.dict())

        return response
>>>>>>> 203c44d (Added working context based worker)
