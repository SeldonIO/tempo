import typing
import tempfile
import os

from typing import Optional, Callable, Any, Dict, get_type_hints, Tuple

from tempo.serve.loader import (
    download,
    upload,
    load_custom,
    save_custom,
    save_environment,
)
from tempo.serve.protocol import Protocol
from tempo.serve.constants import (
    ModelDataType,
    DefaultModelFilename,
    DefaultEnvFilename,
)
from tempo.serve.metadata import (
    ModelDetails,
    ModelDataArgs,
    ModelDataArg,
    ModelFramework,
)


class BaseModel:
    def __init__(
        self,
        name: str,
        user_func: Callable[[Any], Any] = None,
        protocol: Protocol = None,
        local_folder: str = None,
        uri: str = None,
        platform: ModelFramework = None,
        inputs: ModelDataType = None,
        outputs: ModelDataType = None,
    ):
        self._name = name
        self._user_func = user_func
        self.protocol = protocol
        if uri is None:
            uri = ""

        local_folder = self._get_local_folder(local_folder)
        inputs, outputs = self._get_args(inputs, outputs)

        self.details = ModelDetails(
            name=name,
            local_folder=local_folder,
            uri=uri,
            platform=platform,
            inputs=inputs,
            outputs=outputs,
        )

        self.cls = None

    def _get_args(
        self, inputs: ModelDataType = None, outputs: ModelDataType = None
    ) -> Tuple[ModelDataArgs, ModelDataArgs]:
        input_args = []
        output_args = []

        if inputs is None and outputs is None:
            if self._user_func is not None:
                hints = get_type_hints(self._user_func)
                for k, v in hints.items():
                    if k == "return":
                        if isinstance(v, typing._GenericAlias):
                            targs = v.__args__
                            for targ in targs:
                                output_args.append(ModelDataArg(ty=targ))
                        else:
                            output_args.append(ModelDataArg(ty=v))
                    else:
                        input_args.append(ModelDataArg(name=k, ty=v))
        else:
            if type(outputs) == Dict:
                for k, v in outputs.items():
                    output_args.append(ModelDataArg(name=k, ty=v))
            elif type(outputs) == Tuple:
                for ty in list(outputs):
                    output_args.append(ModelDataArg(ty=ty))
            else:
                output_args.append(ModelDataArg(ty=outputs))

            if type(inputs) == Dict:
                for k, v in inputs.items():
                    input_args.append(ModelDataArg(name=k, ty=v))
            elif type(inputs) == Tuple:
                for ty in list(inputs):
                    input_args.append(ModelDataArg(ty=ty))
            else:
                input_args.append(ModelDataArg(ty=inputs))

        return ModelDataArgs(args=input_args), ModelDataArgs(args=output_args)

    def _get_local_folder(self, local_folder: str = None) -> Optional[str]:
        if not local_folder:
            # TODO: Should we do cleanup at some point?
            local_folder = tempfile.mkdtemp()

        return local_folder

    def set_cls(self, cls):
        self.cls = cls

    @classmethod
    def load(cls, file_path: str) -> "BaseModel":
        return load_custom(file_path)

    def save(self, file_path: str = None):
        if not self._user_func:
            # Nothing to save
            return

        if file_path is None:
            file_path = os.path.join(self.details.local_folder, DefaultModelFilename)

        save_custom(self, file_path)

    def upload(self):
        """
        Upload from local folder to uri
        """
        # Save to local_folder before uploading
        self.save()

        # Save environment as well in `local_folder`
        # TODO: Should this be handled in `file_path`?
        file_path = os.path.join(self.details.local_folder, DefaultEnvFilename)
        save_environment(file_path=file_path)

        upload(self.details.local_folder, self.details.uri)

    def download(self):
        """
        Download from uri to local folder
        """
        # TODO: This doesn't make sense for custom methods?
        download(self.details.uri, self.details.local_folder)

    def request(self, req: Dict) -> Dict:
        req_converted = self.protocol.from_protocol_request(req, self.details.inputs)
        if type(req_converted) == dict:
            if self.cls is not None:
                response = self._user_func(self.cls, **req_converted)
            else:
                response = self._user_func(**req_converted)
        elif type(req_converted) == list or type(req_converted) == tuple:
            if self.cls is not None:
                response = self._user_func(self.cls, *req_converted)
            else:
                response = self._user_func(*req_converted)
        else:
            if self.cls is not None:
                response = self._user_func(self.cls, req_converted)
            else:
                response = self._user_func(req_converted)

        if type(response) == dict:
            response_converted = self.protocol.to_protocol_response(
                self.details, **response
            )
        elif type(response) == list or type(response) == tuple:
            response_converted = self.protocol.to_protocol_response(
                self.details, *response
            )
        else:
            response_converted = self.protocol.to_protocol_response(
                self.details, response
            )
        return response_converted
