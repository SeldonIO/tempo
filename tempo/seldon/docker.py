import os
import socket
import time
from typing import Optional, Sequence, Tuple

import docker
from docker.client import DockerClient
from docker.errors import NotFound
from docker.models.containers import Container

from tempo.utils import logger
from tempo.seldon.specs import DefaultHTTPPort, DefaultModelsPath, get_container_spec
from tempo.serve.base import DeployedModel, ModelSpec, Runtime
from tempo.serve.metadata import RuntimeOptions
from tempo.serve.runtime import ModelSpec, Runtime
from tempo.serve.constants import DefaultInsightsServiceName, DefaultInsightsPort, DefaultInsightsImage

DefaultNetworkName = "tempo"


class SeldonDockerRuntime(Runtime):
    def list_models(self) -> Sequence[DeployedModel]:
        pass

    def __init__(self, runtime_options: Optional[RuntimeOptions] = None):
        if runtime_options is None:
            runtime_options = RuntimeOptions()
        runtime_options.runtime = "tempo.seldon.SeldonDockerRuntime"
        super().__init__(runtime_options)

    def _get_host_ip_port(self, model_details: ModelSpec) -> Tuple[str, str]:
        container = self._get_container(model_details)
        port_index = self._get_port_index()
        host_ports = container.ports[port_index]

        host_ip = host_ports[0]["HostIp"]
        host_port = host_ports[0]["HostPort"]
        return host_ip, host_port

    def get_endpoint_spec(self, model_spec: ModelSpec) -> str:
        predict_path = model_spec.protocol.get_predict_path(model_spec.model_details)

        if self._is_inside_docker():
            # If inside Docker, use internal networking
            return f"http://{model_spec.model_details.name}:{DefaultHTTPPort}{predict_path}"

        host_ip, host_port = self._get_host_ip_port(model_spec)

        return f"http://{host_ip}:{host_port}{predict_path}"

    def deploy_spec(self, model_details: ModelSpec):
        try:
            container = self._get_container(model_details)
            if container.status == "running":
                # If container already exists and is running, nothing to do
                # here
                return

            # Remove before re-deploying
            container.remove(force=True)
            self._run_container(model_details)
        except docker.errors.NotFound:
            self._run_container(model_details)

    def deploy_insights_message_dumper(self):
        docker_client = docker.from_env()
        try:
            docker_client.containers.get(DefaultInsightsServiceName)
        except docker.errors.NotFound:
            pass
        else:
            logger.info("Attempted to deploy message dumper but already deployed")
            return
        uid = os.getuid()
        self._create_network(docker_client)
        docker_client.containers.run(
            name=DefaultInsightsServiceName,
            ports={f"{DefaultInsightsPort}/tcp": DefaultInsightsPort},
            image=DefaultInsightsImage,
            detach=True,
            network=DefaultNetworkName,
            user=uid,
        )

    def _run_container(self, model_details: ModelSpec):
        docker_client = docker.from_env()
        uid = os.getuid()

        container_index = self._get_port_index()
        model_folder = model_details.model_details.local_folder
        container_spec = get_container_spec(model_details)
        self._create_network(docker_client)

        docker_client.containers.run(
            name=self._get_container_name(model_details),
            ports={container_index: self._get_available_port()},
            volumes={model_folder: {"bind": DefaultModelsPath, "mode": "ro"}},
            detach=True,
            network=DefaultNetworkName,
            user=uid,
            **container_spec,
        )

    def _create_network(self, docker_client: DockerClient, network_name=DefaultNetworkName):
        try:
            docker_client.networks.get(network_id=network_name)
        except NotFound:
            docker_client.networks.create(name=DefaultNetworkName)

    def _get_port_index(self):
        return f"{DefaultHTTPPort}/tcp"

    def wait_ready_spec(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        host_ip, host_port = self._get_host_ip_port(model_spec)
        ready = False
        t0 = time.time()
        while not ready:
            container = self._get_container(model_spec)
            if container.status == "running":
                # TODO: Use status_path to wait until container is ready
                #  status_path = self.protocol.get_status_path(model_details)
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.connect((host_ip, int(host_port)))
                    return True
                except ConnectionRefusedError:
                    pass
                finally:
                    s.close()
                # url = f"http://{host_ip}:{host_port}{status_path}"
                # print(url)
                # r = requests.get(url)
                # if r.status_code == 200:
                #    return True
            if timeout_secs is not None:
                t1 = time.time()
                if t1 - t0 > timeout_secs:
                    return ready

        return False

    def _get_available_port(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def undeploy_spec(self, model_spec: ModelSpec):
        container = self._get_container(model_spec)
        container.remove(force=True)

    def undeploy_insights_message_dumper(self):
        docker_client = docker.from_env()
        # TODO: Get from constant
        try:
            container = docker_client.containers.get(DefaultInsightsServiceName)
        except docker.errors.NotFound:
            logger.info("Attempted to undeploy insights dumper but container not running")
            return
        container.remove(force=True)

    def _get_container(self, model_details: ModelSpec) -> Container:
        container_name = self._get_container_name(model_details)

        docker_client = docker.from_env()
        container = docker_client.containers.get(container_name)

        return container

    def _get_container_name(self, model_details: ModelSpec):
        return model_details.model_details.name

    def _is_inside_docker(self) -> bool:
        # From https://stackoverflow.com/a/48710609/5015573
        path = "/proc/self/cgroup"
        return os.path.exists("/.dockerenv") or os.path.isfile(path) and any("docker" in line for line in open(path))

    def to_k8s_yaml_spec(self, model_spec: ModelSpec) -> str:
        raise NotImplementedError()
