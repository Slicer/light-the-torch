light-the-torch
===============

.. start-badges

.. list-table::
    :stub-columns: 1

    * - package
      - |license| |status|
    * - code
      - |black| |mypy| |lint|
    * - tests
      - |tests| |coverage|

.. end-badges

With each release of a PyTorch distribution (``torch``, ``torchvision``,
``torchaudio``, ``torchtext``) the wheels are published for combinations of different
Python versions, platforms, and computation backends (CPU, CUDA). Unfortunately, a
differentation based on the computation backend is not supported by
`PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_ . As a workaround the
computation backend is added as a local specifier. For example

.. code-block:: sh

  torch==1.5.1+<computation backend>

Due to this restriction only the wheels of the latest CUDA release are uploaded to
`PyPI <https://pypi.org/search/?q=torch>`_ and thus easily ``pip install`` able. For
other CUDA versions or the installation without CUDA support, one has to resort to
manual version specification:

.. code-block:: sh

  pip install -f https://download.pytorch.org/whl/torch_stable.html torch==1.5.1+<computation backend>

This is especially frustrating if one wants to install packages that depend on one or
several PyTorch distributions: for each package the required PyTorch distributions have
to be manually tracked down, resolved, and installed before the other requirements can
be installed.

``light-the-torch`` offers a small CLI based on ``pip`` to install the PyTorch
distributions from the stable releases. Similar to the Python version and platform, the
computation backend is auto-detected from the available hardware preferring CUDA over
CPU.

Installation
============

The latest **published** version can be installed with

.. code-block:: sh

  pip install light-the-torch


The latest, potentially unstable **development** version can be installed with

.. code-block::

  pip install git+https://github.com/pmeier/light-the-torch

Usage
=====

.. note::

  The following examples were run on a linux machine with Python 3.6 and CUDA 10.1. The
  distributions hosted on PyPI were built with CUDA 10.2.

CLI
---

The CLI of ``light-the-torch`` is invoked with its shorthand ``ltt``

.. code-block:: sh

  $ ltt --help
  usage: ltt [-h] [-V] {install,extract,find} ...

  optional arguments:
    -h, --help            show this help message and exit
    -V, --version         show light-the-torch version and path and exit

  subcommands:
    {install,extract,find}

``ltt install``
^^^^^^^^^^^^^^^

.. code-block:: sh

  $ ltt install --help
  usage: ltt install [-h] [--force-cpu] [--pytorch-only]
                     [--install-cmd INSTALL_CMD] [--verbose]
                     [args [args ...]]

  Install PyTorch distributions from the stable releases. The computation
  backend is auto-detected from the available hardware preferring CUDA over CPU.

  positional arguments:
    args                  arguments of 'pip install'. Optional arguments have to
                          be seperated by '--'

  optional arguments:
    -h, --help            show this help message and exit
    --force-cpu           disable computation backend auto-detection and use CPU
                          instead
    --pytorch-only        install only PyTorch distributions
    --install-cmd INSTALL_CMD
                          installation command for the PyTorch distributions and
                          additional packages. Defaults to 'python -m pip
                          install {packages}'
    --verbose             print more output to STDOUT. For fine control use -v /
                          --verbose and -q / --quiet of the 'pip install'
                          options

``ltt install`` is a drop-in replacement for ``pip install`` without worrying about the
computation backend:

.. code-block:: sh

  $ ltt install torch torchvision
  [...]
  Successfully installed future-0.18.2 numpy-1.19.0 pillow-7.2.0 torch-1.5.1+cu101 torchvision-0.6.1+cu101
  [...]


``ltt install`` is also able to handle packages that depend on PyTorch distributions:

.. code-block:: sh

  $ ltt install kornia
  [...]
  Successfully installed future-0.18.2 numpy-1.19.0 torch-1.5.0+cu101
  [...]
  Successfully installed kornia-0.3.1

``ltt extract``
^^^^^^^^^^^^^^^

.. code-block:: sh

  $ ltt extract --help
  usage: ltt extract [-h] [--verbose] [args [args ...]]

  Extract required PyTorch distributions

  positional arguments:
    args        arguments of 'pip install'. Optional arguments have to be
                seperated by '--'

  optional arguments:
    -h, --help  show this help message and exit
    --verbose   print more output to STDOUT. For fine control use -v / --verbose
                and -q / --quiet of the 'pip install' options


``ltt extract`` extracts the required PyTorch distributions out of packages:

.. code-block:: sh

  $ ltt extract kornia
  torch==1.5.0

.. warning::

  Internally, ``light-the-torch`` uses the ``pip`` resolver which, as of now,
  unfortunately allows conflicting dependencies:

  .. code-block:: sh

    $ ltt extract kornia "torch>1.5"
    torch>1.5

``ltt find``
^^^^^^^^^^^^

.. code-block:: sh

  $ ltt find --help
  usage: ltt find [-h] [--computation-backend COMPUTATION_BACKEND]
                  [--platform PLATFORM] [--python-version PYTHON_VERSION]
                  [--verbose]
                  [args [args ...]]

  Find wheel links for the required PyTorch distributions

  positional arguments:
    args                  arguments of 'pip install'. Optional arguments have to
                          be seperated by '--'

  optional arguments:
    -h, --help            show this help message and exit
    --computation-backend COMPUTATION_BACKEND
                          Only use wheels compatible with COMPUTATION_BACKEND,
                          for example 'cu102' or 'cpu'. Defaults to the
                          computation backend of the running system, preferring
                          CUDA over CPU.
    --platform PLATFORM   Only use wheels compatible with <platform>. Defaults
                          to the platform of the running system.
    --python-version PYTHON_VERSION
                          The Python interpreter version to use for wheel and
                          "Requires-Python" compatibility checks. Defaults to a
                          version derived from the running interpreter. The
                          version can be specified using up to three dot-
                          separated integers (e.g. "3" for 3.0.0, "3.7" for
                          3.7.0, or "3.7.3"). A major-minor version can also be
                          given as a string without dots (e.g. "37" for 3.7.0).
    --verbose             print more output to STDOUT. For fine control use -v /
                          --verbose and -q / --quiet of the 'pip install'
                          options

``ltt find`` finds the links to the wheels of the required PyTorch distributions:

.. code-block:: sh

  $ ltt find torchaudio > requirements.txt
  $ cat requirements.txt
  https://download.pytorch.org/whl/cu101/torch-1.5.1%2Bcu101-cp36-cp36m-linux_x86_64.whl
  https://download.pytorch.org/whl/torchaudio-0.5.1-cp36-cp36m-linux_x86_64.whl

The ``--computation-backend``, ``--platform``, and ``python-version`` options can be
used pin wheel properties instead of auto-detecting them:

.. code-block:: sh

  $ ltt find \
    --computation-backend cu92 \
    --platform win_amd64 \
    --python-version 3.7 \
    torchtext
  https://download.pytorch.org/whl/cu92/torch-1.5.1%2Bcu92-cp37-cp37m-win_amd64.whl
  https://download.pytorch.org/whl/torchtext-0.6.0-py3-none-any.whl


.. note::

  Optional arguments for ``pip install`` have to be passed after a ``--`` seperator.

.. |license|
  image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License

.. |status|
  image:: https://www.repostatus.org/badges/latest/wip.svg
    :alt: Project Status: WIP
    :target: https://www.repostatus.org/#wip

.. |black|
  image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black
   
.. |mypy|
  image:: http://www.mypy-lang.org/static/mypy_badge.svg
    :target: http://mypy-lang.org/
    :alt: mypy

.. |lint|
  image:: https://github.com/pmeier/light-the-torch/workflows/lint/badge.svg
    :target: https://github.com/pmeier/light-the-torch/actions?query=workflow%3Alint+branch%3Amaster
    :alt: Lint status via GitHub Actions

.. |tests|
  image:: https://github.com/pmeier/light-the-torch/workflows/tests/badge.svg
    :target: https://github.com/pmeier/light-the-torch/actions?query=workflow%3Atests+branch%3Amaster
    :alt: Test status via GitHub Actions

.. |coverage|
  image:: https://codecov.io/gh/pmeier/light-the-torch/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pmeier/light-the-torch
    :alt: Test coverage via codecov.io
