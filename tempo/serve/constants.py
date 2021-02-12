from typing import List, Optional, Dict, Type, Union

SKLEARN_MODEL = "sklearn"
XGBOOST_MODEL = "xgboost"
TENSORFLOW_MODEL = "tensorflow"
PYTORCH_MODEL = "pytorch"
ONNX_MODEL = "onnx"

# PyDantic issue doesn't support raw Tuple : https://github.com/samuelcolvin/pydantic/issues/2132
# ModelDataType = Optional[Union[Type,Tuple,Dict[str,Type]]]
ModelDataType = Optional[Union[Type, List, Dict[str, Type]]]

DefaultModelFilename = "model.pickle"
DefaultEnvFilename = "environment.tar.gz"

# TODO: Update once tempo is published
MLServerEnvDeps = ["mlserver==0.3.1.dev1", "mlserver-tempo==0.3.1.dev1"]
