import os

from tempo.serve.metadata import InsightsTypes

CLOUDEVENTS_HEADER_ID = "Ce-Id"
CLOUDEVENTS_HEADER_SPECVERSION = "Ce-Specversion"
CLOUDEVENTS_HEADER_SOURCE = "Ce-Source"
CLOUDEVENTS_HEADER_TYPE = "Ce-Type"
CLOUDEVENTS_HEADER_SPECVERSION_DEFAULT = "0.3"

CLOUDEVENTS_HEADER_REQUEST_ID = "requestid"
CLOUDEVENTS_HEADER_MODEL_ID = "modelid"
CLOUDEVENTS_HEADER_INFERENCE_SERVICE = "inferenceservicename"
CLOUDEVENTS_HEADER_NAMESPACE = "namespace"
CLOUDEVENTS_HEADER_ENDPOINT = "endpoint"

ENV_NAMESPACE = "POD_NAMESPACE"
ENV_SDEP_NAME = "SELDON_DEPLOYMENT_ID"
ENV_PREDICTOR_NAME = "PREDICTOR_ID"
ENV_MODEL_NAME = "PREDICTIVE_UNIT_ID"

NOT_IMPLEMENTED_STR = "NOTIMPLEMENTED"

env_namespace = os.getenv(ENV_NAMESPACE, NOT_IMPLEMENTED_STR)
env_sdep_name = os.getenv(ENV_SDEP_NAME, NOT_IMPLEMENTED_STR)
env_predictor_name = os.getenv(ENV_PREDICTOR_NAME, NOT_IMPLEMENTED_STR)
env_model_name = os.getenv(ENV_MODEL_NAME, NOT_IMPLEMENTED_STR)


# TODO: Add types
def get_cloudevent_headers(request_id: str, ce_type: InsightsTypes):

    ce = {
        CLOUDEVENTS_HEADER_ID: request_id,
        CLOUDEVENTS_HEADER_SPECVERSION: CLOUDEVENTS_HEADER_SPECVERSION_DEFAULT,
        # TODO: Confirm whether we want URL or source - Currently don't have access to URL
        CLOUDEVENTS_HEADER_SOURCE: f"io.seldon.serving.deployment.{env_sdep_name}.{env_namespace}",
        CLOUDEVENTS_HEADER_TYPE: ce_type.value,
        CLOUDEVENTS_HEADER_REQUEST_ID: request_id,
        CLOUDEVENTS_HEADER_MODEL_ID: env_model_name,
        CLOUDEVENTS_HEADER_INFERENCE_SERVICE: env_sdep_name,
        CLOUDEVENTS_HEADER_NAMESPACE: env_namespace,
        CLOUDEVENTS_HEADER_ENDPOINT: env_predictor_name,
    }
    return ce
