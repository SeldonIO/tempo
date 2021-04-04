import pytest

from tempo.serve.metadata import RuntimeOptions


@pytest.mark.parametrize(
    "runtime, replicas",
    [
        ({"k8s_options": {"replicas": 2}}, 2),
        ({}, 1),
    ],
)
def test_runtime_options(runtime, replicas):
    r = RuntimeOptions(**runtime)
    assert r.k8s_options.replicas == replicas
