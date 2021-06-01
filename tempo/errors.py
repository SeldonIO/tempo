class TempoError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class InvalidUserFunction(TempoError):
    def __init__(self, model_name: str, reason: str):
        msg = f"Invalid user function for model {model_name}: {reason}"
        super().__init__(msg)


class UndefinedRuntime(TempoError):
    def __init__(self, model_name: str):
        msg = f"Undefined runtime for model {model_name}"
        super().__init__(msg)


class UndefinedCustomImplementation(TempoError):
    def __init__(self, model_name: str):
        msg = f"Undefined custom implementation for model {model_name}"
        super().__init__(msg)
