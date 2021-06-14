from tempo.seldon.docker import SeldonDockerRuntime
from tempo.serve.model import Model


def test_launch_tensorflow(cifar10_model_deployed_with_runtime: Model, runtime: SeldonDockerRuntime):
    container = runtime._get_container(cifar10_model_deployed_with_runtime.model_spec)
    assert container.status == "running"
