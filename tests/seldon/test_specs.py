from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.seldon.protocol import SeldonProtocol
from tempo.seldon.specs import KubernetesSpec, _V2ContainerFactory, get_container_spec
from tempo.serve.metadata import KubernetesOptions, ModelDataArgs, ModelDetails, ModelFramework, RuntimeOptions
from tempo.serve.model import Model
from tempo.serve.runtime import ModelSpec


def test_kubernetes_spec(sklearn_model: Model):
    k8s_object = KubernetesSpec(sklearn_model.model_spec)

    expected = {
        "apiVersion": "machinelearning.seldon.io/v1",
        "kind": "SeldonDeployment",
        "metadata": {
            "name": sklearn_model.details.name,
            "namespace": sklearn_model.model_spec.runtime_options.k8s_options.namespace,
        },
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

    assert k8s_object.spec == expected


def test_kubernetes_spec_pipeline():
    details = ModelDetails(
        name="inference-pipeline",
        platform=ModelFramework.TempoPipeline,
        uri="gs://seldon/tempo",
        local_folder="",
        inputs=ModelDataArgs(args=[]),
        outputs=ModelDataArgs(args=[]),
    )
    options = KubernetesOptions(namespace="production", replicas=1)
    protocol = KFServingV2Protocol()
    runtime_options = RuntimeOptions(k8s_options=options)
    model_spec = ModelSpec(model_details=details, protocol=protocol, runtime_options=runtime_options)
    k8s_object = KubernetesSpec(model_spec)

    container_spec = _V2ContainerFactory.get_container_spec(details, runtime_options)
    container_env = [{"name": name, "value": value} for name, value in container_spec["environment"].items()]

    expected = {
        "apiVersion": "machinelearning.seldon.io/v1",
        "kind": "SeldonDeployment",
        "metadata": {"name": details.name, "namespace": options.namespace},
        "spec": {
            "protocol": "kfserving",
            "predictors": [
                {
                    "componentSpecs": [
                        {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "inference-pipeline",
                                        "image": container_spec["image"],
                                        "env": container_env,
                                        "args": [],
                                    }
                                ]
                            }
                        }
                    ],
                    "graph": {
                        "modelUri": details.uri,
                        "name": "inference-pipeline",
                        "type": "MODEL",
                        "implementation": "TRITON_SERVER",
                        "serviceAccountName": "tempo-pipeline",
                    },
                    "name": "default",
                    "replicas": options.replicas,
                }
            ],
        },
    }

    assert k8s_object.spec == expected


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
