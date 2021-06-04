import json
import os
from types import SimpleNamespace

from google.protobuf.json_format import ParseDict
from protoc_gen_validate.validator import validate
from seldon_deploy_sdk import ApiClient, V1PredictionSchema

from tempo.metadata.prediction_schema_pb2 import PredictionSchema

TESTS_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(TESTS_PATH, "data")


class DummyResponse:
    def __init__(self, data: str):
        self.data = data


def test_sdk_deserialize():
    api_client = ApiClient()
    with open(DATA_PATH + "/income-classifier.json", "r") as f:
        raw = f.read()
        res = SimpleNamespace(data=raw)
        cls = api_client.deserialize(res, V1PredictionSchema)
        print(cls)


def test_parse_income_classifier():
    with open(DATA_PATH + "/income-classifier.json", "r") as f:
        raw = f.read()
        j = json.loads(raw)
        ps = PredictionSchema()
        ParseDict(j, ps)
        validate(ps)
        print(ps)


def test_parse_cifar10():
    with open(DATA_PATH + "/cifar10-images.json", "r") as f:
        raw = f.read()
        j = json.loads(raw)
        ps = PredictionSchema()
        ParseDict(j, ps)
        validate(ps)
        print(ps)
