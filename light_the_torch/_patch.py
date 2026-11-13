import contextlib
import dataclasses
import enum
import functools
import inspect
import itertools
import optparse
import os
import re
import sys
import unittest.mock
from typing import List, Set
from unittest import mock

import pip._internal.cli.cmdoptions
from pip._internal.index.collector import CollectedSources
from pip._internal.index.package_finder import CandidateEvaluator
from pip._internal.index.sources import build_source
from pip._internal.models.search_scope import SearchScope
from pip._vendor.packaging.utils import canonicalize_name

import light_the_torch as ltt

from . import _cb as cb
from ._utils import apply_fn_patch

# pip>=24.0 requires the project name to be passed to build_source()
BUILD_SOURCE_REQUIRES_PROJECT_NAME = (
    "project_name" in inspect.signature(build_source).parameters
)


class Channel(enum.Enum):
    STABLE = enum.auto()
    TEST = enum.auto()
    NIGHTLY = enum.auto()

    @classmethod
    def from_str(cls, string):
        return cls[string.upper()]


PYTORCH_DISTRIBUTIONS = {
    "torch",
    "torch-model-archiver",
    "torch-tb-profiler",
    "torcharrow",
    "torchaudio",
    "torchcsprng",
    "torchdata",
    "torchdistx",
    "torchserve",
    "torchtext",
    "torchvision",
    "pytorch-triton",
}

THIRD_PARTY_PACKAGES = {
    "Jinja2",
    "MarkupSafe",
    "Pillow",
    "certifi",
    "charset-normalizer",
    "cmake",
    "colorama",
    "filelock",
    "fsspec",
    "idna",
    "lit",
    "mpmath",
    "networkx",
    "numpy",
    "nvidia-cublas-cu11",
    "nvidia-cuda-cupti-cu11",
    "nvidia-cuda-nvrtc-cu11",
    "nvidia-cuda-runtime-cu11",
    "nvidia-cudnn-cu11",
    "nvidia-cufft-cu11",
    "nvidia-curand-cu11",
    "nvidia-cusolver-cu11",
    "nvidia-cusparse-cu11",
    "nvidia-nccl-cu11",
    "nvidia-nvtx-cu11",
    "packaging",
    "portalocker",
    "requests",
    "sympy",
    "tqdm",
    "typing-extensions",
    "urllib3",
}

# Dependencies of ROCm builds of PyTorch, e.g. triton-rocm, rocm, rocm-sdk-core,
# rocm-sdk-libraries, and rocm-sdk-device-gfx1201. They are only available on (or
# linked from) the PyTorch indices. PyPI only hosts outdated versions or placeholders
# with the same names, which cannot satisfy the requirements of PyTorch.
ROCM_PACKAGES = {
    "pytorch-triton-rocm",
    "triton-rocm",
    "rocm",
    "rocm-profiler",
}
ROCM_PACKAGE_PREFIXES = ("rocm-sdk-",)


def is_rocm_package(project_name):
    name = canonicalize_name(project_name)
    return name in ROCM_PACKAGES or name.startswith(ROCM_PACKAGE_PREFIXES)


def patch(pip_main):
    @functools.wraps(pip_main)
    def wrapper(argv=None):
        if argv is None:
            argv = sys.argv[1:]

        with apply_patches(argv):
            return pip_main(argv)

    return wrapper


# adapted from https://stackoverflow.com/a/9307174
class PassThroughOptionParser(optparse.OptionParser):
    def __init__(self):
        super().__init__(add_help_option=False)

    def _process_args(self, largs, rargs, values):
        while rargs:
            try:
                super()._process_args(largs, rargs, values)
            except (optparse.BadOptionError, optparse.AmbiguousOptionError) as error:
                largs.append(error.opt_str)


@dataclasses.dataclass
class LttOptions:
    computation_backends: Set[cb.ComputationBackend] = dataclasses.field(
        default_factory=lambda: {cb.CPUBackend()}
    )
    channel: Channel = Channel.STABLE

    @staticmethod
    def computation_backend_parser_options():
        return [
            optparse.Option(
                "--pytorch-computation-backend",
                help=(
                    "Computation backend for compiled PyTorch distributions, "
                    "e.g. 'cu118', 'cu121', or 'cpu'. "
                    "Multiple computation backends can be passed as a comma-separated "
                    "list, e.g 'cu118,cu121'. "
                    "If not specified, the computation backend is detected from the "
                    "available hardware, preferring CUDA over CPU."
                ),
            ),
            optparse.Option(
                "--cpuonly",
                action="store_true",
                help=(
                    "Shortcut for '--pytorch-computation-backend=cpu'. "
                    "If '--computation-backend' is used simultaneously, "
                    "it takes precedence over '--cpuonly'."
                ),
            ),
        ]

    @staticmethod
    def channel_parser_option() -> optparse.Option:
        return optparse.Option(
            "--pytorch-channel",
            help=(
                "Channel to download PyTorch distributions from, e.g. 'stable' , "
                "'test', and 'nightly'. "
                "If not specified, defaults to 'stable' unless '--pre' is given in "
                "which case it defaults to 'test'."
            ),
        )

    @staticmethod
    def _parse(argv):
        parser = PassThroughOptionParser()

        for option in LttOptions.computation_backend_parser_options():
            parser.add_option(option)
        parser.add_option(LttOptions.channel_parser_option())
        parser.add_option("--pre", dest="pre", action="store_true")

        opts, _ = parser.parse_args(argv)
        return opts

    @classmethod
    def from_pip_argv(cls, argv: List[str]):
        if not argv or argv[0] != "install":
            return cls()

        opts = cls._parse(argv)

        if opts.pytorch_computation_backend is not None:
            cbs = {
                cb.ComputationBackend.from_str(string.strip())
                for string in opts.pytorch_computation_backend.split(",")
            }
        elif opts.cpuonly:
            cbs = {cb.CPUBackend()}
        elif "LTT_PYTORCH_COMPUTATION_BACKEND" in os.environ:
            cbs = {
                cb.ComputationBackend.from_str(string.strip())
                for string in os.environ["LTT_PYTORCH_COMPUTATION_BACKEND"].split(",")
            }
        else:
            cbs = cb.detect_compatible_computation_backends()

        if opts.pytorch_channel is not None:
            channel = Channel.from_str(opts.pytorch_channel)
        elif opts.pre:
            channel = Channel.TEST
        else:
            channel = Channel.STABLE

        return cls(cbs, channel)


@contextlib.contextmanager
def apply_patches(argv):
    options = LttOptions.from_pip_argv(argv)

    patches = [
        patch_cli_version(),
        patch_cli_options(),
        patch_link_collection_with_supply_chain_attack_mitigation(
            options.computation_backends, options.channel
        ),
        patch_candidate_selection(options.computation_backends),
    ]

    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)

        yield stack


@contextlib.contextmanager
def patch_cli_version():
    with apply_fn_patch(
        "pip",
        "_internal",
        "cli",
        "main_parser",
        "get_pip_version",
        postprocessing=lambda input, output: f"ltt {ltt.__version__} from {ltt.__path__[0]}\n{output}",
    ):
        yield


@contextlib.contextmanager
def patch_cli_options():
    def postprocessing(input, output):
        for option in LttOptions.computation_backend_parser_options():
            input.cmd_opts.add_option(option)

    index_group = pip._internal.cli.cmdoptions.index_group

    with apply_fn_patch(
        "pip",
        "_internal",
        "cli",
        "cmdoptions",
        "add_target_python_options",
        postprocessing=postprocessing,
    ):
        with unittest.mock.patch.dict(index_group):
            options = index_group["options"].copy()
            options.append(LttOptions.channel_parser_option)
            index_group["options"] = options
            yield


def get_index_urls(computation_backends, channel):
    if channel == Channel.STABLE:
        channel_paths = [""]
    else:
        channel_paths = [f"{channel.name.lower()}/"]
    return [
        f"https://download.pytorch.org/whl/{channel_path}{backend}"
        for channel_path, backend in itertools.product(
            channel_paths, sorted(computation_backends)
        )
    ]


@contextlib.contextmanager
def patch_link_collection_with_supply_chain_attack_mitigation(
    computations_backends, channel
):
    def is_pinned(requirement):
        if requirement.req is None:
            return False

        return requirement.is_pinned

    @contextlib.contextmanager
    def context(input):
        with patch_link_collection(
            computations_backends,
            channel,
            {
                requirement.name
                for requirement in input.root_reqs
                if requirement.user_supplied and is_pinned(requirement)
            },
        ):
            yield

    with apply_fn_patch(
        "pip",
        "_internal",
        "resolution",
        "resolvelib",
        "resolver",
        "Resolver",
        "resolve",
        context=context,
    ):
        yield


@contextlib.contextmanager
def patch_link_collection(computation_backends, channel, user_supplied_pinned_packages):
    search_scope = SearchScope(
        find_links=[],
        index_urls=get_index_urls(computation_backends, channel),
        no_index=False,
    )

    rocm_selected = any(
        isinstance(computation_backend, cb.ROCmBackend)
        for computation_backend in computation_backends
    )

    @contextlib.contextmanager
    def context(input):
        if not (
            input.project_name in PYTORCH_DISTRIBUTIONS
            or (
                input.project_name in THIRD_PARTY_PACKAGES
                and input.project_name not in user_supplied_pinned_packages
            )
            or (rocm_selected and is_rocm_package(input.project_name))
        ):
            yield
            return

        with mock.patch.object(input.self, "search_scope", search_scope):
            yield

    def postprocessing(input, output):
        if input.project_name not in PYTORCH_DISTRIBUTIONS:
            return output

        if channel != Channel.STABLE:
            return output

        # Some stable binaries are not hosted on the PyTorch indices. We check if this
        # is the case for the current distribution.
        for remote_file_source in output.index_urls:
            candidates = list(remote_file_source.page_candidates())

            # Cache the candidates, so `pip` doesn't has to retrieve them again later.
            remote_file_source.page_candidates = lambda: iter(candidates)

            # If there are any candidates on the PyTorch indices, we continue normally.
            if candidates:
                return output

        # In case the distribution is not present on the PyTorch indices, we fall back
        # to PyPI.
        build_source_kwargs = dict(
            candidates_from_page=input.candidates_from_page,
            page_validator=input.self.session.is_secure_origin,
            expand_dir=False,
            cache_link_parsing=False,
        )
        if BUILD_SOURCE_REQUIRES_PROJECT_NAME:
            build_source_kwargs["project_name"] = input.project_name
        _, pypi_file_source = build_source(
            SearchScope(
                find_links=[],
                index_urls=["https://pypi.org/simple"],
                no_index=False,
            ).get_index_urls_locations(input.project_name)[0],
            **build_source_kwargs,
        )

        return CollectedSources(find_links=[], index_urls=[pypi_file_source])

    with apply_fn_patch(
        "pip",
        "_internal",
        "index",
        "collector",
        "LinkCollector",
        "collect_sources",
        context=context,
        postprocessing=postprocessing,
    ):
        yield


@contextlib.contextmanager
def patch_candidate_selection(computation_backends):
    computation_backend_link_pattern = re.compile(
        r"/(?P<computation_backend>(cpu|cu\d+|rocm([\d.]+)))/"
    )

    def extract_local_specifier(candidate):
        local = candidate.version.local

        # Make sure that local actually is a computation backend identifier
        if local is not None:
            try:
                cb.ComputationBackend.from_str(local)
            except ValueError:
                local = None

        if local is None:
            match = computation_backend_link_pattern.search(candidate.link.comes_from)
            local = match["computation_backend"] if match else "any"

        # Early PyTorch distributions used the "any" local specifier to indicate a
        # pure Python binary. This was changed to no local specifier later.
        # Setting this to "cpu" is technically not correct as it will exclude this
        # binary if a non-CPU backend is requested. Still, this is probably the
        # right thing to do, since the user requested a specific backend and
        # although this binary will work with it, it was not compiled against it.
        if local == "any":
            local = "cpu"

        return local

    def preprocessing(input):
        if not input.candidates:
            return

        candidates = iter(input.candidates)
        candidate = next(candidates)

        if candidate.name not in PYTORCH_DISTRIBUTIONS:
            # At this stage all candidates have the same name. Thus, if the first is
            # not a PyTorch distribution, we don't need to check the rest and can
            # return without changes.
            return

        input.candidates = [
            candidate
            for candidate in itertools.chain([candidate], candidates)
            if extract_local_specifier(candidate) in computation_backends
        ]

    vanilla_sort_key = CandidateEvaluator._sort_key

    def patched_sort_key(candidate_evaluator, candidate):
        # At this stage all candidates have the same name. Thus, we don't need to
        # mirror the exact key structure that the vanilla sort keys have.
        return (
            vanilla_sort_key(candidate_evaluator, candidate)
            if candidate.name not in PYTORCH_DISTRIBUTIONS
            else (
                cb.ComputationBackend.from_str(extract_local_specifier(candidate)),
                candidate.version.base_version,
            )
        )

    with apply_fn_patch(
        "pip",
        "_internal",
        "index",
        "package_finder",
        "CandidateEvaluator",
        "get_applicable_candidates",
        preprocessing=preprocessing,
    ):
        with unittest.mock.patch.object(
            CandidateEvaluator, "_sort_key", new=patched_sort_key
        ):
            yield
