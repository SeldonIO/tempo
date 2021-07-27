from tempo.seldon.protocol import SeldonProtocol
from tempo.seldon.specs import KubernetesSpec, get_container_spec
from tempo.serve.base import ModelSpec
from tempo.serve.metadata import KubernetesRuntimeOptions, ModelDataArgs, ModelDetails, ModelFramework


def test_kubernetes_spec(sklearn_model):

    k8s_object = KubernetesSpec(sklearn_model.model_spec, KubernetesRuntimeOptions())

    expected = {
        "spec": {
            "protocol": "seldon",
            "predictors": [
                {
                    "annotations": {"seldon.io/no-engine": "true"},
                    "graph": {
                        "modelUri": sklearn_model.details.uri,
                        "name": "test-iris-sklearn",
                        "type": "MODEL",
                        "implementation": KubernetesSpec.Implementations[sklearn_model.details.platform],
                    },
                    "name": "default",
                    "replicas": sklearn_model.model_spec.runtime_options.replicas,
                }
            ],
        },
    }

    assert k8s_object.spec["spec"] == expected["spec"]


def test_tensorflow_spec():
    md = ModelDetails(
        name="test",
        local_folder="",
        uri="",
        platform=ModelFramework.Tensorflow,
        inputs=ModelDataArgs(args=[]),
        outputs=ModelDataArgs(args=[]),
    )
    protocol = SeldonProtocol()
    runtime_options = KubernetesRuntimeOptions(namespace="production", replicas=1)
    model_spec = ModelSpec(model_details=md, protocol=protocol, runtime_options=runtime_options)
    spec = get_container_spec(model_spec)
    assert "image" in spec
    assert "command" in spec
