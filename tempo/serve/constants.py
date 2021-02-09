from typing import List, Optional, Dict, Type, Union, Tuple

SKLEARN_MODEL = "sklearn"
XGBOOST_MODEL = "xgboost"
TENSORFLOW_MODEL = "tensorflow"
PYTORCH_MODEL = "pytorch"
ONNX_MODEL = "onnx"

# PyDantic issue doesn't support raw Tuple : https://github.com/samuelcolvin/pydantic/issues/2132
# ModelDataType = Optional[Union[Type,Tuple,Dict[str,Type]]]
ModelDataType = Optional[Union[Type, List, Dict[str, Type]]]
