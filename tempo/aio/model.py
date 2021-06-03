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
    async def _session(self) -> aiohttp.ClientSession:
        if self._client_session is None:
            # TODO: Delete at some point
            self._client_session = aiohttp.ClientSession(raise_for_status=True)

        return self._client_session

    async def remote(self, *args, **kwargs):
        # TODO: Decouple to support multiple transports (e.g. Kafka, gRPC)
        model_spec = self._get_model_spec()
        remoter = self._create_remote(model_spec)
        prot = model_spec.protocol
        ingress_options = model_spec.runtime_options.ingress_options

        req = prot.to_protocol_request(*args, **kwargs)
        endpoint = remoter.get_endpoint_spec(model_spec)
        headers = remoter.get_headers(model_spec)
        response_raw = await self._session.post(
            endpoint, json=req, headers=headers, verify_ssl=ingress_options.verify_ssl
        )

        response_json = await response_raw.json()
        output_schema = model_spec.model_details.outputs
        return prot.from_protocol_response(response_json, output_schema)

    async def __call__(self, *args, **kwargs) -> Any:
        future = super().__call__(*args, **kwargs)
        return await future
