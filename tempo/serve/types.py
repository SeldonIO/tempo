from typing import Any, Callable, Dict, List, Optional, Type, Union

# PyDantic issue doesn't support raw Tuple : https://github.com/samuelcolvin/pydantic/issues/2132
# ModelDataType = Optional[Union[Type,Tuple,Dict[str,Type]]]
ModelDataType = Optional[Union[Type, List, Dict[str, Type]]]
PredictMethodSignature = Callable[..., Any]
LoadMethodSignature = Callable[[], None]
