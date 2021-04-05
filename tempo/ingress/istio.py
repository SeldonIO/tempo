from kubernetes import client, config

from tempo.serve.ingress import Ingress
from tempo.serve.runtime import ModelSpec


class IstioIngress(Ingress):
    def get_external_host_url(self, model_spec: ModelSpec) -> str:
        config.load_kube_config()
        api_instance = client.CoreV1Api()
        res = api_instance.list_namespaced_service("istio-system", field_selector="metadata.name=istio-ingressgateway")
        ingress_ip = res.items[0].status.load_balancer.ingress[0].ip
        if not ingress_ip:
            ingress_ip = res.items[0].status.load_balancer.ingress[0].hostname
        scheme = "http"
        if model_spec.runtime_options.ingress_options.ssl:
            scheme = "https"
        return f"{scheme}://{ingress_ip}"
