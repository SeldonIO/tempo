import numpy as np

from tempo.serve.utils import _get_predict_method


def test_get_predict_method(inference_pipeline_class):
    predict_method = _get_predict_method(inference_pipeline_class.__class__)

    assert predict_method == inference_pipeline_class.__class__.p


def test_bind_init(inference_pipeline_class):
    user_func = inference_pipeline_class.pipeline._user_func

    assert user_func.__self__ == inference_pipeline_class
    assert user_func == inference_pipeline_class.p


def test_K_reference(inference_pipeline_class):
    MyClass = inference_pipeline_class.__class__

    assert inference_pipeline_class.pipeline._K == MyClass


def test_multiple_instances(inference_pipeline_class):
    MyClass = inference_pipeline_class.__class__
    inference_pipeline_class_2 = MyClass()

    payload = np.array([0, 1, 2, 3])

    self_1 = inference_pipeline_class.pipeline._user_func.__self__
    self_2 = inference_pipeline_class_2.pipeline._user_func.__self__
    assert self_1 != self_2
    assert self_1 == inference_pipeline_class
    assert self_2 == inference_pipeline_class_2

    inference_pipeline_class(payload=payload)
    assert inference_pipeline_class.counter == 1
    assert inference_pipeline_class_2.counter == 0

    inference_pipeline_class_2(payload=payload)
    assert inference_pipeline_class.counter == 1
    assert inference_pipeline_class_2.counter == 1

    inference_pipeline_class(payload=payload)
    assert inference_pipeline_class.counter == 2
    assert inference_pipeline_class_2.counter == 1
