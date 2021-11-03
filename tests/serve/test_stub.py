import json

import numpy as np

from tempo.protocols.v2 import V2Protocol
from tempo.serve.base import ModelSpec
from tempo.serve.metadata import KFServingOptions, ModelDataArg, ModelDataArgs, ModelDetails, ModelFramework


def test_model_data_arg():

    m = ModelDataArg(ty=str)
    s = m.json()
    j = json.loads(s)
    assert j["ty"] == "builtins.str"

    m = ModelDataArg(ty=np.ndarray)
    s = m.json()
    j = json.loads(s)
    assert j["ty"] == "numpy.ndarray"

    m = ModelDataArg(**j)


def test_model_data_args():
    args = ModelDataArgs(args=[ModelDataArg(ty=str)])
    s = args.json()
    j = json.loads(s)
    assert j["args"][0]["ty"] == "builtins.str"


def test_model_spec():
    ms = ModelSpec(
        model_details=ModelDetails(
            name="test",
            local_folder="",
            uri="",
            platform=ModelFramework.XGBoost,
            inputs=ModelDataArgs(args=[ModelDataArg(ty=str)]),
            outputs=ModelDataArgs(args=[]),
        ),
        protocol=V2Protocol(),
        runtime_options=KFServingOptions().local_options,
    )
    s = ms.json()
    j = json.loads(s)
    ms2 = ModelSpec(**j)
    assert isinstance(ms2.protocol, V2Protocol)
    assert ms2.model_details.inputs.args[0].ty == str
