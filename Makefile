SHELL := /bin/bash 
VERSION = $(shell sed 's/^__version__ = "\(.*\)"/\1/' ./tempo/version.py)

.PHONY: install
install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt

.PHONY: test
test: tempo
	pytest tempo

fmt:
	black ./ --exclude "(.eggs|.tox)"


.PHONY: lint
lint:
	flake8 .

.PHONY: mypy
mypy:
	mypy .

install-rclone:
	curl https://rclone.org/install.sh | sudo bash

.PHONY: tempo/tests/examples
tempo/tests/examples:
	mkdir -p tempo/tests/examples
	cd tempo/tests/examples && \
		gsutil cp -r gs://seldon-models/sklearn . && \
		gsutil cp -r gs://seldon-models/xgboost . && \
		gsutil cp -r gs://seldon-models/mlflow . && \
		gsutil cp -r gs://seldon-models/keras . && \
		gsutil cp -r gs://seldon-models/tfserving .


clean_test_data:
	rm -rf tempo/tests/examples


build: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf ./dist ./build *.egg-info

push-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

push:
	twine upload dist/*


version:
	@echo ${VERSION}
