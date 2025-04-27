# Contributing guide lines

Thanks a lot for your interest to contribute to `light-the-torch`! All contributions are
appreciated, be it code or not. Especially in a project like this, we rely on user
reports for edge cases we didn't anticipate. Please feel free to
[open an issue](<(https://github.com/Slicer/light-the-torch/issues)>) if you encounter
anything that you think should be working but doesn't.

If you are planning to contribute bug-fixes or documentation improvements, please go
ahead and open a [pull request (PR)](https://github.com/Slicer/light-the-torch/pulls).
If you are planning to contribute new features, please open an
[issue](https://github.com/Slicer/light-the-torch/issues) and discuss the feature with
us first.

## Workflow

To start working on `light-the-torch` clone the repository from GitHub and set up the
development environment

```shell
git clone https://github.com/Slicer/light-the-torch
cd light-the-torch
virtualenv .venv --prompt='(light-the-torch-dev) '
source .venv/bin/activate
pip install doit
doit install
```

Every PR is subjected to multiple checks that it has to pass before it can be merged.
The checks are performed through [doit](https://pydoit.org/). Below you can find details
and instructions how to run the checks locally.

### Code format and linting

`light-the-torch` uses [ufmt](https://ufmt.omnilib.dev/en/stable/) to format Python
code, and [flake8](https://flake8.pycqa.org/en/stable/) to enforce
[PEP8](https://www.python.org/dev/peps/pep-0008/) compliance.

To automatically format the code, run

```shell
doit format
```

Instead of running the formatting manually, you can also add
[pre-commit](https://pre-commit.com/) hooks. By running

```shell
pre-commit install
```

once, an equivalent of `doit format` is run everytime you `git commit` something.

Everything that cannot be fixed automatically, can be checked with

```shell
doit lint
```

### Tests

`light-the-torch` uses [pytest](https://docs.pytest.org/en/stable/) to run the test
suite. You can run it locally with

```sh
doit test
```
