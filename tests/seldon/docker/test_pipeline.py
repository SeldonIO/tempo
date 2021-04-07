import docker
import numpy as np
import pytest
import yaml

from tempo.seldon.docker import SeldonDockerRuntime
from tempo.serve.constants import MLServerEnvDeps
from tempo.serve.pipeline import Pipeline


def test_conda_yaml(pipeline_conda_yaml):
    print(pipeline_conda_yaml)
    with open(pipeline_conda_yaml) as f:
        env = yaml.safe_load(f)
        for dep in env["dependencies"]:
            if dep == "pip":
                assert dep[0] == MLServerEnvDeps[0]


def test_deploy_pipeline_docker(
    inference_pipeline: Pipeline,
    runtime: SeldonDockerRuntime,
):
    for model in inference_pipeline.models.values():
        container = runtime._get_container(model.model_spec)
        assert container.status == "running"

    pipeline_container = runtime._get_container(inference_pipeline.model_spec)
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
def test_pipeline_remote(inference_pipeline: Pipeline, runtime: SeldonDockerRuntime, x_input):
    y_pred = inference_pipeline.remote(payload=x_input)

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
def test_seldon_pipeline_request_docker(inference_pipeline: Pipeline, x_input, expected):
    y_pred = inference_pipeline.request(x_input)

    assert y_pred == expected


def test_undeploy_pipeline_docker(inference_pipeline: Pipeline, runtime: SeldonDockerRuntime):
    runtime.undeploy(inference_pipeline)

    for model in inference_pipeline.models.values():
        with pytest.raises(docker.errors.NotFound):
            runtime._get_container(model.model_spec)
