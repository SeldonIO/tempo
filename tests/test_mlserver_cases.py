from mlserver.settings import ModelParameters, ModelSettings

from tempo import Model
from tempo.serve.loader import save


def case_custom_model(custom_model: Model) -> ModelSettings:
    save(custom_model, save_env=False)
    model_uri = custom_model.details.local_folder

    return ModelSettings(
        name="custom-model",
        parameters=ModelParameters(uri=model_uri),
    )


def case_wrapped_class_instance(inference_pipeline_class) -> ModelSettings:
    save(inference_pipeline_class, save_env=False)
    model_uri = inference_pipeline_class.pipeline.details.local_folder

    return ModelSettings(
        name="wrapped-class-instance",
        parameters=ModelParameters(uri=model_uri),
    )


def case_wrapped_class(inference_pipeline_class) -> ModelSettings:
    MyClass = inference_pipeline_class.__class__

    save(MyClass, save_env=False)
    model_uri = MyClass.pipeline.details.local_folder

    return ModelSettings(
        name="wrapped-class",
        parameters=ModelParameters(uri=model_uri),
    )
