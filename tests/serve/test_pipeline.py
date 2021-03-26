from typing import Tuple

from tempo.serve.utils import pipeline, predictmethod


def test_class_func_class():
    @pipeline(
        name="classifier",
        models=[],
    )
    class MyPipeline:
        @predictmethod
        def predict(self, X: str) -> str:
            return X

    x = MyPipeline()

    r = x.predict("hello")
    assert r == "hello"
    r = x("hello")
    assert r == "hello"


def test_class_func():
    @pipeline(
        name="classifier",
        models=[],
    )
    def predict(X: str) -> str:
        return X

    r = predict("hello")
    assert r == "hello"


def test_clear_state_func():
    @pipeline(
        name="classifier",
        models=[],
    )
    class MyPipeline:
        def __init__(self):
            self.cleared = False

        @predictmethod
        def predict(self, X: str) -> str:
            return X

    x = MyPipeline()

    @pipeline(
        name="classifier",
        models=[x],
    )
    class MyPipeline2:
        def __init__(self):
            self.cleared = False

        @predictmethod
        def predict(self, X: str) -> str:
            return x(X=X)

    y = MyPipeline2()

    y(X="hello")


def test_class_two_outputs():
    @pipeline(
        name="classifier",
        models=[],
    )
    class MyPipeline:
        @predictmethod
        def predict(self, X: str) -> Tuple[str, str]:
            return X, X

    x = MyPipeline()

    r1, r2 = x.predict("hello")
    assert r1 == "hello"
    assert r2 == "hello"
