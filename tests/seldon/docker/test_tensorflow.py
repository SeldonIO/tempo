from tempo.seldon.docker import SeldonDockerRuntime
from tempo.serve.model import Model


def test_launch_tensorflow(cifar10_model: Model, runtime: SeldonDockerRuntime):
    container = runtime._get_container(cifar10_model.model_spec)
    assert container.status == "running"
