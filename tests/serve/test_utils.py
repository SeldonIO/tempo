from tempo.serve.utils import _get_funcs


def test_get_funcs(inference_pipeline_class):
    predict_func, load_func = _get_funcs(inference_pipeline_class.__class__)

    assert predict_func == inference_pipeline_class.__class__.p
    assert load_func is None


def test_bind_init(inference_pipeline_class):
    user_func = inference_pipeline_class.pipeline._user_func

    assert user_func.__self__ == inference_pipeline_class
    assert user_func == inference_pipeline_class.p
