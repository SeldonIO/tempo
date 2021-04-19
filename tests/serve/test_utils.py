from tempo.serve.utils import _get_predict_method


def test_get_predict_method(inference_pipeline_class):
    predict_method = _get_predict_method(inference_pipeline_class.__class__)

    assert predict_method == inference_pipeline_class.__class__.p


def test_bind_init(inference_pipeline_class):
    user_func = inference_pipeline_class.pipeline._user_func

    assert user_func.__self__ == inference_pipeline_class
    assert user_func == inference_pipeline_class.p
