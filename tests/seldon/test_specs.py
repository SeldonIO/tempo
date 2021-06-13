from tempo.seldon.protocol import SeldonProtocol
from tempo.seldon.specs import KubernetesSpec, get_container_spec
from tempo.serve.base import ModelSpec
from tempo.serve.metadata import KubernetesOptions, ModelDataArgs, ModelDetails, ModelFramework, RuntimeOptions


def test_kubernetes_spec(sklearn_model):
    k8s_object = KubernetesSpec(sklearn_model.model_spec)

    expected = {
        "spec": {
            "protocol": "seldon",
            "predictors": [
                {
                    "graph": {
                        "modelUri": sklearn_model.details.uri,
                        "name": "test-iris-sklearn",
                        "type": "MODEL",
                        "implementation": KubernetesSpec.Implementations[sklearn_model.details.platform],
                    },
                    "name": "default",
                    "replicas": sklearn_model.model_spec.runtime_options.k8s_options.replicas,
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
    options = KubernetesOptions(namespace="production", replicas=1)
    runtime_options = RuntimeOptions(k8s_options=options)
    model_spec = ModelSpec(model_details=md, protocol=protocol, runtime_options=runtime_options)
    spec = get_container_spec(model_spec)
    assert "image" in spec
    assert "command" in spec
