import numpy as np
import pytest
import yaml

from tempo.seldon.docker import SeldonDockerRuntime
from tempo.serve.base import ClientModel
from tempo.serve.constants import MLServerEnvDeps
from tempo.serve.pipeline import Pipeline


def test_conda_yaml(pipeline_conda_yaml):
    with open(pipeline_conda_yaml) as f:
        env = yaml.safe_load(f)
        for dep in env["dependencies"]:
            # we want to fetch the pip dependencies
            if isinstance(dep, dict):
                assert dep["pip"][0] == MLServerEnvDeps[0]


def test_deploy_pipeline_docker(
    inference_pipeline_deployed_with_runtime,
    runtime: SeldonDockerRuntime,
):
    for model in inference_pipeline_deployed_with_runtime.models.values():
        container = runtime._get_container(model.model_spec)
        assert container.status == "running"

    pipeline_container = runtime._get_container(inference_pipeline_deployed_with_runtime.model_spec)
    assert pipeline_container.status == "running"


@pytest.mark.parametrize(
    "x_input",
    [
        pytest.param(
            [[1, 2, 3, 4]],
            marks=pytest.mark.skip(reason="not supported with KFServingV2Protocol"),
        ),
        np.array([[0.5, 2, 3, 4]]),
        pytest.param(
            {
                "inputs": [
                    {
                        "name": "payload",
                        "shape": [1, 4],
                        "datatype": "INT32",
                        "data": [[0.4, 2, 3, 4]],
                    }
                ]
            },
            marks=pytest.mark.skip(reason="not supported with KFServingV2Protocol"),
        ),
    ],
)
def test_pipeline_remote(inference_pipeline_deployed: ClientModel, x_input):
    y_pred = inference_pipeline_deployed.predict(payload=x_input)

    np.testing.assert_allclose(y_pred, [2.0], atol=1e-2)


@pytest.mark.parametrize(
    "x_input, expected",
    [
        (
            {
                "inputs": [
                    {
                        "name": "input0",
                        "datatype": "FP64",
                        "shape": [1, 4],
                        "data": [0.4, 2, 3, 4],
                    }
                ]
            },
            {
                "model_name": "inference-pipeline",
                "outputs": [{"name": "output0", "datatype": "FP64", "shape": [1], "data": [2.0]}],
            },
        )
    ],
)
def test_seldon_pipeline_request_docker(inference_pipeline_deployed_with_runtime: Pipeline, x_input, expected):
    y_pred = inference_pipeline_deployed_with_runtime.request(x_input)

    assert y_pred == expected
