# `light-the-torch`

[![BSD-3-Clause License](https://img.shields.io/github/license/Slicer/light-the-torch)](https://opensource.org/licenses/BSD-3-Clause)
[![Project Status: WIP](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Code coverage via codecov.io](https://codecov.io/gh/Slicer/light-the-torch/branch/main/graph/badge.svg)](https://codecov.io/gh/Slicer/light-the-torch)

`light-the-torch` is a small utility that wraps `pip` to ease the installation process
for PyTorch distributions like `torch`, `torchvision`, `torchaudio`, and so on as well
as third-party packages that depend on them. It auto-detects compatible CUDA versions
from the local setup and installs the correct PyTorch binaries without user
interference.

- [Why do I need it?](#why-do-i-need-it)
- [How do I install it?](#how-do-i-install-it)
- [How do I use it?](#how-do-i-use-it)
- [How does it work?](#how-does-it-work)
- [Is it safe?](#is-it-safe)
- [How do I contribute?](#how-do-i-contribute)

## Why do I need it?

PyTorch distributions like `torch`, `torchvision`, `torchaudio`, and so on are fully
`pip install`'able, but PyPI, the default `pip` search index, has some limitations:

1. PyPI regularly only allows binaries up to a size of
   [approximately 60 MB](https://github.com/pypa/packaging-problems/issues/86). One can
   [request a file size limit increase](https://pypi.org/help/#file-size-limit) (and the
   PyTorch team probably does that for every release), but it is still not enough:
   although PyTorch has pre-built binaries for Windows with CUDA, they cannot be
   installed through PyPI due to their size.
2. PyTorch uses local version specifiers to indicate for which computation backend the
   binary was compiled, for example `torch==1.11.0+cpu`. Unfortunately, local specifiers
   are not allowed on PyPI. Thus, only the binaries compiled with one CUDA version are
   uploaded without an indication of the CUDA version. If you do not have a CUDA capable
   GPU, downloading this is only a waste of bandwidth and disk capacity. If on the other
   hand your NVIDIA driver version simply doesn't support the CUDA version the binary
   was compiled with, you can't use any of the GPU features.

To overcome this, PyTorch also hosts _most_[^1] binaries
[on their own package indices](https://download.pytorch.org/whl). To access PyTorch's
package indices, you can still use `pip install`, but some
[additional options](https://pytorch.org/get-started/locally/) are needed:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

[^1]:
    Some distributions are not compiled against a specific computation backend and thus
    hosting them on PyPI is sufficient since they work in every environment.

While this is certainly an improvement, it still has a few downsides:

1. You need to know what computation backend, e.g. CUDA 11.8 (`cu118`), is supported on
   your local machine. This can be quite challenging for new users and at least tedious
   for more experienced ones.
2. Besides the stable binaries, PyTorch also offers nightly and test ones. To install
   them, you need a different `--index-url` for each.

If any of these points don't sound appealing to you, and you just want to have the same
user experience as `pip install` for PyTorch distributions, `light-the-torch` was made
for you.

## How do I install it?

Installing `light-the-torch` is as easy as

```shell
pip install light-the-torch
```

Since it depends on `pip` and it might be upgraded during installation,
[Windows users](https://pip.pypa.io/en/stable/installation/#upgrading-pip) should
install it with

```shell
py -m pip install light-the-torch
```

## How do I use it?

After `light-the-torch` is installed you can use its CLI interface `ltt` as drop-in
replacement for `pip`:

```shell
ltt install torch
```

In fact, `ltt` is `pip` with a few added options:

- By default, `ltt` uses the local NVIDIA driver version to select the correct binary
  for you. You can pass the `--pytorch-computation-backend` option to manually specify
  the computation backend you want to use:

  ```shell
  ltt install --pytorch-computation-backend=cu121 torch torchvision torchaudio
  ```

  Borrowing from the mutex packages that PyTorch provides for `conda` installations,
  `--cpuonly` is available as shorthand for `--pytorch-computation-backend=cpu`.

  In addition, the computation backend to be installed can also be set through the
  `LTT_PYTORCH_COMPUTATION_BACKEND` environment variable. It will only be honored in
  case no CLI option for the computation backend is specified.

- By default, `ltt` installs stable PyTorch binaries. To install binaries from the
  nightly or test channels pass the `--pytorch-channel` option:

  ```shell
  ltt install --pytorch-channel=nightly torch torchvision torchaudio
  ```

  If `--pytorch-channel` is not passed, using `pip`'s builtin `--pre` option implies
  `--pytorch-channel=test`.

Of course, you are not limited to install only PyTorch distributions. Everything shown
above also works if you install packages that depend on PyTorch:

```shell
ltt install --pytorch-computation-backend=cpu --pytorch-channel=nightly pystiche
```

## How does it work?

The authors of `pip` **do not condone** the use of `pip` internals as they might break
without warning. As a results of this, `pip` has no capability for plugins to hook into
specific tasks.

`light-the-torch` works by monkey-patching `pip` internals at runtime:

- While searching for a download link for a PyTorch distribution, `light-the-torch`
  replaces the default search index with an official PyTorch download link. This is
  equivalent to calling `pip install` with the `--index-url` option only for PyTorch
  distributions.
- While evaluating possible PyTorch installation candidates, `light-the-torch` culls
  binaries incompatible with the hardware.

## Is it safe?

A project as large as PyTorch is attractive for malicious actors given the large user
base. For example in December 2022, PyTorch was hit by a
[supply chain attack](https://pytorch.org/blog/compromised-nightly-dependency/) that
potentially extracted user information. The PyTorch team mitigated the attack as soon as
it was detected by temporarily hosting all third party dependencies on their own
indices. With that,
`pip install torch --extra-index-url https://download.pytorch.org/whl/cpu` wouldn't pull
anything from PyPI and thus avoiding malicious packages placed there. Ultimately, this
became the permanent solution and the official installation instructions now use
`--index-url` and thus preventing installing anything not hosted on their indices.

However, due to `light-the-torch`'s index patching, this mitigation was initially
completely circumvented since only PyTorch distributions would have been installed from
the PyTorch indices. Since version `0.7.0`, `light-the-torch` will only pull third-party
dependencies from PyPI in case they are specifically requested and pinned. For example
`ltt install --pytorch-channel=nightly torch` and
`ltt install --pytorch-channel=nightly torch sympy` will install everything from the
PyTorch indices. However, if you pin a third party dependency, e.g.
`ltt install --pytorch-channel=nightly torch sympy==1.11.1`, it will be pulled from PyPI
regardless of whether the version matches the one on the PyTorch index.

In summary, `light-the-torch` is usually as safe as the regular PyTorch installation
instructions. However, attacks on the supply chain can lead to situations where
`light-the-torch` circumvents mitigations done by the PyTorch team. Unfortunately,
`light-the-torch` is not officially supported by PyTorch and thus also not tested by
them.

## How do I contribute?

Thanks a lot for your interest to contribute to `light-the-torch`! All contributions are
appreciated, be it code or not. Especially in a project like this, we rely on user
reports for edge cases we didn't anticipate. Please feel free to
[open an issue](https://github.com/Slicer/light-the-torch/issues) if you encounter
anything that you think should be working but doesn't.

If you want to contribute code, check out our [contributing guidelines](CONTRIBUTING.md)
to learn more about the workflow.
