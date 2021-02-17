import os

from typing import Dict
from setuptools import find_packages, setup


ROOT_PATH = os.path.dirname(__file__)
PKG_NAME = "tempo"
PKG_PATH = os.path.join(ROOT_PATH, PKG_NAME)


def _load_version() -> str:
    version = ""
    version_path = os.path.join(PKG_PATH, "version.py")
    with open(version_path) as fp:
        version_module: Dict[str, str] = {}
        exec(fp.read(), version_module)
        version = version_module["__version__"]

    return version


def _load_description() -> str:
    readme_path = os.path.join(ROOT_PATH, "README.md")
    with open(readme_path) as fp:
        return fp.read()


setup(
    #  name=PKG_NAME,
    # TODO: Update once we've got consensus on package name
    name="mlops-tempo",
    author="Seldon Technologies Ltd.",
    author_email="hello@seldon.io",
    version=_load_version(),
    description="Machine Learning Operations Toolkit",
    url="https://github.com/SeldonIO/tempo",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    setup_requires=["pytest-runner"],
    install_requires=[
        "attrs",
        "numpy",
        "kubernetes",
        "docker",
        "packaging",
        "requests",
        "pydantic",
        "cloudpickle",
        "python-rclone",
        "seldon-deploy-sdk",
        "conda-pack",
        # TODO: Remove `tensorflow` package coming from protos
        "tensorflow",
    ],
    tests_require=["pytest", "pytest-cov", "pytest-xdist", "pytest-lazy-fixture"],
    zip_safe=False,
    long_description=_load_description(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
)
