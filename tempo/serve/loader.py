import cloudpickle


def save_custom(pipeline, file_path: str) -> str:
    with open(file_path, "wb") as file:
        cloudpickle.dump(pipeline, file)

    return file_path


def load_custom(file_path: str):
    with open(file_path, "rb") as file:
        return cloudpickle.load(file)
