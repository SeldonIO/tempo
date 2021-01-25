from setuptools import find_packages, setup
from tempo import __version__


def readme():
    with open("README.md") as f:
        return f.read()


# read version file
exec(open("tempo/version.py").read())


setup(
    name="tempo",
    author="Seldon Technologies Ltd.",
    author_email="hello@seldon.io",
    version=__version__,  # type: ignore # noqa F821
    description="Machine Learning Operations Toolkit",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SeldonIO/tempo",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    setup_requires=["pytest-runner"],
    install_requires=[
        "numpy",
        "scikit-learn",
        "tensorflow>=2.0",
        "xgboost",
        "mlflow",
        "azure-common==1.1.25",
        "azure-core==1.8.0",
        "azure-storage-blob==12.4.0",
        "azure-storage-common==2.1.0",
        "minio",
        "pysimplegui",
        "seldon-core",
        "kubernetes",
        "packaging",
        "seldon-deploy-sdk",
        "requests",
        "pydantic",
        "cloudpickle",
    ],
    tests_require=["pytest", "pytest-cov", "pytest-xdist", "pytest-lazy-fixture"],
    zip_safe=False,
)
