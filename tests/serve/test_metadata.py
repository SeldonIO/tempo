import pytest

from tempo.serve.metadata import KubernetesRuntimeOptions


@pytest.mark.parametrize(
    "runtime, replicas",
    [
        ({"replicas": 2}, 2),
        ({}, 1),
    ],
)
def test_runtime_options(runtime, replicas):
    r = KubernetesRuntimeOptions(**runtime)
    assert r.replicas == replicas
