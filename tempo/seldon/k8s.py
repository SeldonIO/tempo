import json
import os
import time
from typing import Optional, Sequence

import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from tempo.k8s.constants import TempoK8sLabel, TempoK8sModelSpecAnnotation
from tempo.seldon.endpoint import Endpoint
from tempo.seldon.specs import KubernetesSpec
from tempo.serve.base import DeployedModel, ModelSpec, Runtime
from tempo.serve.constants import ENV_K8S_SERVICE_HOST
from tempo.serve.metadata import RuntimeOptions
from tempo.serve.stub import deserialize
from tempo.utils import logger
from tempo.serve.constants import DefaultInsightsServiceName, DefaultInsightsPort, DefaultInsightsImage, DefaultSeldonSystemNamespace


class SeldonCoreOptions(RuntimeOptions):
    runtime: str = "tempo.seldon.SeldonKubernetesRuntime"


class SeldonKubernetesRuntime(Runtime):
    def __init__(self, runtime_options: Optional[RuntimeOptions] = None):
        if runtime_options is None:
            runtime_options = RuntimeOptions()
        runtime_options.runtime = "tempo.seldon.SeldonKubernetesRuntime"
        super().__init__(runtime_options)

    def create_k8s_client(self):
        inside_cluster = os.getenv(ENV_K8S_SERVICE_HOST)
        if inside_cluster:
            logger.debug("Loading cluster local config")
            config.load_incluster_config()
        else:
            logger.debug("Loading external kubernetes config")
            config.load_kube_config()

    def get_endpoint_spec(self, model_spec: ModelSpec) -> str:
        self.create_k8s_client()
        endpoint = Endpoint()
        return endpoint.get_url(model_spec)

    def undeploy_spec(self, model_spec: ModelSpec):
        self.create_k8s_client()
        api_instance = client.CustomObjectsApi()
        api_instance.delete_namespaced_custom_object(
            "machinelearning.seldon.io",
            "v1",
            model_spec.runtime_options.k8s_options.namespace,
            "seldondeployments",
            model_spec.model_details.name,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )

    def undeploy_insights_message_dumper(self):
        self.create_k8s_client()
        api_instance = client.CoreV1Api()
        api_instance.delete_namespaced_pod(
            DefaultInsightsServiceName,
            DefaultSeldonSystemNamespace)
        api_instance.delete_namespaced_service(
            DefaultInsightsServiceName,
            DefaultSeldonSystemNamespace)

    def deploy_insights_message_dumper(self):
        self.create_k8s_client()
        api_instance = client.CoreV1Api()

        try:
            existing = api_instance.read_namespaced_pod(
                DefaultInsightsServiceName,
                DefaultSeldonSystemNamespace,
            )
        except ApiException as e:
            if e.status != 404:
                raise e

        k8s_pod_spec = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": DefaultInsightsServiceName,
                "labels": { "app": DefaultInsightsServiceName, },
            },
            "spec": {
                "containers": [ {
                    "name": "default",
                    "image": DefaultInsightsImage,
                    "ports": [ { "containerPort": DefaultInsightsPort } ]
                } ]
            }
        }
        logger.debug(f"Creating kubernetes insights manager pod: {k8s_pod_spec}")
        api_instance.create_namespaced_pod(
            DefaultSeldonSystemNamespace,
            k8s_pod_spec)

        k8s_svc_spec = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": DefaultInsightsServiceName,
            },
            "spec": {
                "selector": { "app": DefaultInsightsServiceName },
                "ports": [ {
                    "port": DefaultInsightsPort,
                    "targetPort": DefaultInsightsPort,
                } ],
            }
        }
        logger.debug(f"Creating kubernetes insights manager pod: {k8s_svc_spec}")
        api_instance.create_namespaced_service(
            DefaultSeldonSystemNamespace,
            k8s_svc_spec)

    def deploy_spec(self, model_spec: ModelSpec):
        self.create_k8s_client()
        k8s_specer = KubernetesSpec(model_spec)
        k8s_spec = k8s_specer.spec
        logger.debug(k8s_spec)

        api_instance = client.CustomObjectsApi()

        try:
            existing = api_instance.get_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                model_spec.runtime_options.k8s_options.namespace,
                "seldondeployments",
                model_spec.model_details.name,
            )
            k8s_spec["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
            api_instance.replace_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                model_spec.runtime_options.k8s_options.namespace,
                "seldondeployments",
                model_spec.model_details.name,
                k8s_spec,
            )
        except ApiException as e:
            if e.status == 404:
                api_instance.create_namespaced_custom_object(
                    "machinelearning.seldon.io",
                    "v1",
                    model_spec.runtime_options.k8s_options.namespace,
                    "seldondeployments",
                    k8s_spec,
                )
            else:
                raise e

    def wait_ready_spec(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        self.create_k8s_client()
        ready = False
        t0 = time.time()
        while not ready:
            api_instance = client.CustomObjectsApi()
            existing = api_instance.get_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                model_spec.runtime_options.k8s_options.namespace,
                "seldondeployments",
                model_spec.model_details.name,
            )
            if "status" in existing and "state" in existing["status"]:
                ready = existing["status"]["state"] == "Available"
            if timeout_secs is not None:
                t1 = time.time()
                if t1 - t0 > timeout_secs:
                    return ready
        return ready

    def to_k8s_yaml_spec(self, model_spec: ModelSpec) -> str:
        k8s_spec = KubernetesSpec(model_spec)
        return yaml.safe_dump(k8s_spec.spec)

    def list_models(self, namespace: Optional[str] = None) -> Sequence[DeployedModel]:
        self.create_k8s_client()
        api_instance = client.CustomObjectsApi()

        if namespace is None and self.runtime_options is not None:
            namespace = self.runtime_options.k8s_options.namespace

        if namespace is None:
            return []

        try:
            models = []
            response = api_instance.list_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1",
                namespace=namespace,
                plural="seldondeployments",
                label_selector=TempoK8sLabel + "=true",
            )
            for model in response["items"]:
                metadata = model["metadata"]["annotations"][TempoK8sModelSpecAnnotation]
                remote_model = deserialize(json.loads(metadata))
                models.append(remote_model)
            return models
        except ApiException as e:
            if e.status == 404:
                return []
            else:
                raise e
