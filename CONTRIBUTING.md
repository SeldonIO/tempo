# Contributing to Tempo

_Before opening a pull request_ consider:

- Is the change important and ready enough to ask the community to spend time reviewing?
- Have you searched for existing, related issues and pull requests?
- Is the change being proposed clearly explained and motivated?

When you contribute code, you affirm that the contribution is your original work and that you
license the work to the project under the project's open source license. Whether or not you
state this explicitly, by submitting any copyrighted material via pull request, email, or
other means you agree to license the material under the project's open source license and
warrant that you have the legal authority to do so.

## Release notes

Our process to manage release notes is modelled after how the Kubernetes project handles them.
This process can be separated into 2 separate phases: 

- Adding notes on each PR. Happens at **PR creation time**.
- Compiling all PR notes before a release. Happens at **release time**.

## Coding conventions

We use [pre-commit](https://pre-commit.com/) to handle a number of Git hooks
which ensure that all changes to the codebase follow Seldon's code conventions.
It is recommended to set these up before making any change to the codebase.
Extra checks down the line will stop the build if the code is not compliant to
the style guide of each language in the repository.

To install it, follow the [official instructions](https://pre-commit.com/#install).
Once installed, run:

```console
$ pre-commit install
```

This will read the hooks defined in `.pre-commit-config.yaml` and install them
accordingly on your local repository.


## Tests

Regardless of the package you are working on, we abstract the main tasks to a
`Makefile`.
Therefore, in order to run the tests, you should be able to just do:

```bash
$ make test
```

### Python

We use [pytest](https://docs.pytest.org/en/latest/) as our main test runner.
However, to ensure that tests run on the same version of the package that final
users will download from `pip` and pypi.org, we use
[tox](https://tox.readthedocs.io/en/latest/) on top of it.
To install both (plus other required plugins), just run:

```bash
$ make install_dev
```

Using `tox` we can run the entire test suite over different environments,
isolated between them.
You can see the different ones we currently use on the
[setup.cfg](https://github.com/SeldonIO/seldon-core/blob/master/python/setup.cfg)
file.
You can run your tests across all these environments using the standard `make test` [mentioned above](#Tests).
Alternatively, if you want to pass any extra parameters, you can also run `tox`
directly as:

```bash
$ tox
```

One of the caveats of `tox` is that, as the number of environments grows, so
does the time it takes to finish running the tests.
As a solution, during local development it may be recommended to run `pytest` directly
on your own environment.
You can do so as:

```bash
$ pytest
```

### Additional system requirements for running the tests:

- `docker`
- `conda`: for example `Miniconda`, follow the steps described in this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/).
- Models artifacts:
For first time, install `gsutil` and download the models from Seldon's Google Storage bucket:
```bash
$ sudo make tests/testdata
```
