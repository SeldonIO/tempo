from __future__ import annotations
import abc
import attr
from tempo.serve.metadata import ModelDetails


@attr.s(auto_attribs=True)
class Runtime(abc.ABC):
    #TODO change to deploy_model
    @abc.abstractmethod
    def deploy(self, model_details: ModelDetails):
        pass

    #TODO change to undeploy_model
    @abc.abstractmethod
    def undeploy(self, model_details: ModelDetails):
        pass

    @abc.abstractmethod
    def get_endpoint(self, model_details: ModelDetails):
        pass

    @abc.abstractmethod
    def wait_ready(self, model_details: ModelDetails, timeout_secs=None) -> bool:
        pass

    #@abc.abstractmethod
    #def predict_model(self, model_details:ModelDetails, request: dict):
    #    pass

    # @abc.abstractmethod
    # def predict_pipeline(self, request: dict):
    #    pass

    @abc.abstractmethod
    def get_protocol(self):
        pass

    #TODO change to to_yaml
    @abc.abstractmethod
    def to_k8s_yaml(self, model_details: ModelDetails) -> str:
        pass

    #@abc.abstractmethod
    #def deploy_pipeline(self):
    #    pass

    #@abc.abstractmethod
    #def undeploy_pipeline(self):
    #    pass

    #TODO add predict_pipeline
    #OR add separate model_runtime and pipeline_runtime interfaces?