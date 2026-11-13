from unittest import mock

import pytest

from light_the_torch import _cb as cb, _patch
from pip._internal.index.collector import LinkCollector
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession


@pytest.mark.parametrize(
    "project_name",
    [
        "rocm",
        "rocm-profiler",
        "rocm-sdk-core",
        "rocm-sdk-libraries",
        "rocm-sdk-devel",
        "rocm-sdk-device-gfx1201",
        "rocm_sdk_device_gfx90a",
        "triton-rocm",
        "pytorch-triton-rocm",
        "Triton_ROCm",
    ],
)
def test_is_rocm_package(project_name):
    assert _patch.is_rocm_package(project_name)


@pytest.mark.parametrize(
    "project_name", ["torch", "triton", "pytorch-triton", "numpy", "rocmlike"]
)
def test_is_not_rocm_package(project_name):
    assert not _patch.is_rocm_package(project_name)


def collect_index_urls(computation_backends, project_name):
    link_collector = LinkCollector(
        session=PipSession(),
        search_scope=SearchScope.create(
            find_links=[], index_urls=["https://pypi.org/simple"], no_index=False
        ),
    )
    with _patch.patch_link_collection(
        computation_backends, _patch.Channel.STABLE, set()
    ):
        sources = link_collector.collect_sources(
            project_name, candidates_from_page=lambda link: []
        )
    return [source.link.url for source in sources.index_urls]


@pytest.mark.parametrize(
    "project_name", ["triton-rocm", "rocm", "rocm-sdk-core", "rocm-sdk-device-gfx1201"]
)
def test_rocm_packages_from_pytorch_index(project_name):
    index_urls = collect_index_urls(
        {cb.ROCmBackend(7, 2), cb.CPUBackend()}, project_name
    )

    assert index_urls
    assert all(
        url.startswith("https://download.pytorch.org/whl/") for url in index_urls
    )
    assert any("/whl/rocm7.2/" in url for url in index_urls)


@pytest.mark.parametrize(
    "computation_backends",
    [{cb.CPUBackend()}, {cb.CUDABackend(12, 8), cb.CPUBackend()}],
    ids=["cpu", "cuda"],
)
def test_rocm_packages_from_pypi_without_rocm_backend(computation_backends):
    index_urls = collect_index_urls(computation_backends, "triton-rocm")

    assert index_urls == ["https://pypi.org/simple/triton-rocm/"]


def test_other_packages_unaffected_by_rocm_backend():
    index_urls = collect_index_urls({cb.ROCmBackend(7, 2)}, "requests-oauthlib")

    assert index_urls == ["https://pypi.org/simple/requests-oauthlib/"]


def test_fallback_to_pypi_if_not_on_pytorch_index():
    # Simulate that the PyTorch indices have no distributions for the project. This is
    # the code path that failed with pip>=24.0, because build_source() requires the
    # project name since then.
    with mock.patch(
        "pip._internal.index.sources._RemoteFileSource.page_candidates",
        return_value=iter([]),
    ):
        index_urls = collect_index_urls({cb.ROCmBackend(10, 1)}, "torch")

    assert index_urls == ["https://pypi.org/simple/torch/"]
