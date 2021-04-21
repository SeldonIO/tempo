import os
from types import FunctionType
from typing import Any

import dill

from ..constants import DefaultModelFilename, DefaultRemoteFilename


def _fixed_create_function(
    fcode,
    fglobals,
    fname=None,
    fdefaults=None,
    fclosure=None,
    fdict=None,
    fkwdefaults=None,
):
    """
    Patched version of `dill`'s `_create_function`, to make sure __builtins__
    is serialised.

    Patched version from:
        https://github.com/uqfoundation/dill/issues/219#issuecomment-522068603
    Original source code can be found at:
        https://github.com/uqfoundation/dill/blob/846b888ef957dc1991464b562df425aa35142cb8/dill/_dill.py#L594-L603

    """
    # same as FunctionType, but enable passing __dict__ to new function,
    # __dict__ is the storehouse for attributes added after function creation
    if fdict is None:
        fdict = dict()
    func = FunctionType(fcode, fglobals or dict(), fname, fdefaults, fclosure)
    func.__dict__.update(fdict)  # XXX: better copy? option to copy?
    if fkwdefaults is not None:
        func.__kwdefaults__ = fkwdefaults

    # THE WORKAROUND:
    # if the function was serialized without recurse, fglobals would actually contain
    # __builtins__, but because of recurse only the referenced modules/objects
    # end up in fglobals and we are missing the important __builtins__
    if "__builtins__" not in func.__globals__:
        func.__globals__["__builtins__"] = globals()["__builtins__"]

    return func


def _patch_dill(f):
    def _patched(*args, **kwargs):
        _dill = dill._dill
        # Save original before patching
        original_create_function = _dill._create_function
        _dill._create_function = _fixed_create_function

        ret = f(*args, **kwargs)

        _dill._create_function = original_create_function

        return ret

    return _patched


@_patch_dill
def save(tempo_artifact: Any, save_env=True):
    model = tempo_artifact.get_tempo()
    model.save(save_env=save_env)


@_patch_dill
def save_custom(pipeline, file_path: str) -> str:
    with open(file_path, "wb") as file:
        dill.dump(pipeline, file, recurse=True)

    return file_path


def load(folder: str):
    file_path_pkl = os.path.join(folder, DefaultModelFilename)
    return load_custom(file_path_pkl)


def load_custom(file_path: str):
    with open(file_path, "rb") as file:
        return dill.load(file)


def load_remote(folder: str):
    file_path = os.path.join(folder, DefaultRemoteFilename)
    with open(file_path, "rb") as file:
        return dill.load(file)
