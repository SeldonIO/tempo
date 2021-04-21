from typing import Any, Callable, Tuple, get_type_hints

from .metadata import ModelDataArg, ModelDataArgs
from .types import ModelDataType


def infer_args(func: Callable[..., Any]) -> Tuple[ModelDataArgs, ModelDataArgs]:
    hints = get_type_hints(func)
    input_args = []
    output_args = []

    for k, v in hints.items():
        if k == "return":
            if hasattr(v, "__args__"):
                # NOTE: If `__args__` are present, assume this as a
                # `typing.Generic`, like `Tuple`
                targs = v.__args__
                for targ in targs:
                    output_args.append(ModelDataArg(ty=targ))
            else:
                output_args.append(ModelDataArg(ty=v))
        else:
            input_args.append(ModelDataArg(name=k, ty=v))

    return ModelDataArgs(args=input_args), ModelDataArgs(args=output_args)


def process_datatypes(inputs: ModelDataType, outputs: ModelDataType) -> Tuple[ModelDataArgs, ModelDataArgs]:
    input_args = _process_datatypes(datatypes=inputs)
    output_args = _process_datatypes(datatypes=outputs)

    return input_args, output_args


def _process_datatypes(datatypes: ModelDataType) -> ModelDataArgs:
    args = []

    if isinstance(datatypes, dict):
        for k, v in datatypes.items():
            args.append(ModelDataArg(name=k, ty=v))
    elif isinstance(datatypes, tuple):
        for ty in list(datatypes):
            args.append(ModelDataArg(ty=ty))
    else:
        args.append(ModelDataArg(ty=datatypes))

    return ModelDataArgs(args=args)
