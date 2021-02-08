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
        "kubernetes",
        "packaging",
        "requests",
        "pydantic",
        "cloudpickle",
        "python-rclone",
        "seldon-deploy-sdk"
    ],
    tests_require=["pytest", "pytest-cov", "pytest-xdist", "pytest-lazy-fixture"],
    zip_safe=False,
)
