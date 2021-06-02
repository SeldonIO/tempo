import aiohttp

from typing import Any
from inspect import iscoroutinefunction

from ..serve.model import Model as _Model
from ..errors import InvalidUserFunction


class Model(_Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        is_async = iscoroutinefunction(self._user_func)
        if not is_async:
            raise InvalidUserFunction(self._name, reason="function is not awaitable")

        self._client_session = None

    @property
    async def _session(self) -> aiohttp.Session:
        if self._client_session is None:
            # TODO: Delete at some point
            self._client_session = aiohttp.ClientSession()

        return self._client_session

    async def remote(self, *args, **kwargs):
        model_spec = self._get_model_spec()
        remoter = self._create_remote(model_spec)

        prot = model_spec.protocol
        req = prot.to_protocol_request(*args, **kwargs)
        endpoint = remoter.get_endpoint_spec(model_spec)
        response_raw = await self._session.post(endpoint, json=req)

        response_json = await response_raw.json()
        output_schema = model_spec.model_details.outputs
        return prot.from_protocol_response(response_json, output_schema)

    async def __call__(self, *args, **kwargs) -> Any:
        future = super().__call__(*args, **kwargs)
        return await future
