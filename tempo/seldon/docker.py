import docker
import socket
import requests
import os
import time

from typing import Any

from docker.models.containers import Container
from docker.client import DockerClient
from docker.errors import NotFound

from tempo.serve.protocol import Protocol
from tempo.serve.runtime import Runtime
from tempo.serve.metadata import ModelDetails
from tempo.seldon.protocol import SeldonProtocol
from tempo.seldon.specs import (
    DefaultHTTPPort,
    DefaultModelsPath,
    get_container_spec,
)

DefaultNetworkName = "tempo"


class SeldonDockerRuntime(Runtime):
    def __init__(self, protocol=None):
        if protocol is None:
            self.protocol = SeldonProtocol()
        else:
            self.protocol = protocol

    def get_protocol(self) -> Protocol:
        return self.protocol

    def _get_host_ip_port(self, model_details: ModelDetails) -> (str, str):
        container = self._get_container(model_details)
        port_index = self._get_port_index()
        host_ports = container.ports[port_index]

        host_ip = host_ports[0]["HostIp"]
        host_port = host_ports[0]["HostPort"]
        return host_ip, host_port

    def get_endpoint(self, model_details: ModelDetails) -> str:
        protocol = self.get_protocol()
        predict_path = protocol.get_predict_path(model_details)

        if self._is_inside_docker():
            # If inside Docker, use internal networking
            return f"http://{model_details.name}:{DefaultHTTPPort}{predict_path}"

        host_ip, host_port = self._get_host_ip_port(model_details)

        return f"http://{host_ip}:{host_port}{predict_path}"

    def remote(self, model_details: ModelDetails, *args, **kwargs) -> Any:
        protocol = self.get_protocol()
        req = protocol.to_protocol_request(*args, **kwargs)
        endpoint = self.get_endpoint(model_details)
        response_raw = requests.post(endpoint, json=req)
        return protocol.from_protocol_response(
            response_raw.json(), model_details.outputs
        )

    def deploy(self, model_details: ModelDetails):
        docker_client = docker.from_env()

        try:
            self._get_container(model_details)
        except docker.errors.NotFound:
            protocol = self.get_protocol()
            container_index = self._get_port_index()
            model_folder = model_details.local_folder
            container_spec = get_container_spec(model_details, protocol)
            self._create_network(docker_client)

            docker_client.containers.run(
                name=self._get_container_name(model_details),
                ports={container_index: self._get_available_port()},
                volumes={model_folder: {"bind": DefaultModelsPath, "mode": "ro"}},
                detach=True,
                network=DefaultNetworkName,
                **container_spec,
            )

    def _create_network(
        self, docker_client: DockerClient, network_name=DefaultNetworkName
    ):
        try:
            docker_client.networks.get(network_id=network_name)
        except NotFound:
            docker_client.networks.create(name=DefaultNetworkName)

    def _get_port_index(self):
        return f"{DefaultHTTPPort}/tcp"

    def wait_ready(self, model_details: ModelDetails, timeout_secs=None) -> bool:
        host_ip, host_port = self._get_host_ip_port(model_details)
        ready = False
        t0 = time.time()
        while not ready:
            container = self._get_container(model_details)
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

    def _get_available_port(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def undeploy(self, model_details: ModelDetails):
        container = self._get_container(model_details)
        container.remove(force=True)

    def _get_container(self, model_details: ModelDetails) -> Container:
        container_name = self._get_container_name(model_details)

        docker_client = docker.from_env()
        container = docker_client.containers.get(container_name)

        return container

    def _get_container_name(self, model_details: ModelDetails):
        return model_details.name

    def _is_inside_docker(self) -> bool:
        # From https://stackoverflow.com/a/48710609/5015573
        path = "/proc/self/cgroup"
        return (
            os.path.exists("/.dockerenv")
            or os.path.isfile(path)
            and any("docker" in line for line in open(path))
        )

    def to_k8s_yaml(self, model_details: ModelDetails) -> str:
        raise NotImplementedError()
