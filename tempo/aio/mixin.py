from inspect import iscoroutinefunction
from typing import Any, Dict

import aiohttp

from tempo.serve.base import ModelSpec

from ..errors import InvalidUserFunction, UndefinedCustomImplementation


class _AsyncMixin:
    def __init__(self):
        if self._user_func is not None:
            is_async = iscoroutinefunction(self._user_func)
            if not is_async:
                raise InvalidUserFunction(
                    self.model_spec.model_details.name,
                    reason="function is not awaitable",
                )

        self._client_session = None

    @property
    def _session(self) -> aiohttp.ClientSession:
        if self._client_session is None:
            # TODO: Delete at some point
            self._client_session = aiohttp.ClientSession(raise_for_status=True)

        return self._client_session

    async def remote_with_spec(self, model_spec: ModelSpec, *args, **kwargs):
        remoter = self._create_remote(model_spec)  # type: ignore
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

    async def predict(self, *args, **kwargs):
        # TODO: Decouple to support multiple transports (e.g. Kafka, gRPC)
        model_spec = self._get_model_spec(None)
        return await self.remote_with_spec(model_spec, *args, **kwargs)

    async def __call__(self, *args, **kwargs) -> Any:
        future = super().__call__(*args, **kwargs)  # type: ignore
        return await future

    async def request(self, req: Dict) -> Dict:
        # TODO: Decouple to avoid duplicating code
        if self._user_func is None:  # type: ignore
            raise UndefinedCustomImplementation(self.details.name)  # type: ignore

        prot = self.model_spec.protocol  # type: ignore
        req_converted = prot.from_protocol_request(req, self.details.inputs)  # type: ignore

        if type(req_converted) == dict:
            response = await self(**req_converted)
        elif type(req_converted) == list or type(req_converted) == tuple:
            response = await self(*req_converted)
        else:
            response = await self(req_converted)

        if type(response) == dict:
            response_converted = prot.to_protocol_response(self.details, **response)  # type: ignore
        elif type(response) == list or type(response) == tuple:
            response_converted = prot.to_protocol_response(self.details, *response)  # type: ignore
        else:
            response_converted = prot.to_protocol_response(self.details, response)  # type: ignore

        return response_converted
