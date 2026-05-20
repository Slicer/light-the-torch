import subprocess

from types import SimpleNamespace

import pytest

from light_the_torch import _cb as cb

try:
    subprocess.check_call(
        "nvidia-smi",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    NVIDIA_DRIVER_AVAILABLE = True
except subprocess.CalledProcessError:
    NVIDIA_DRIVER_AVAILABLE = False


skip_if_nvidia_driver_unavailable = pytest.mark.skipif(
    not NVIDIA_DRIVER_AVAILABLE, reason="Requires nVidia driver."
)


class GenericComputationBackend(cb.ComputationBackend):
    @property
    def local_specifier(self):
        return "generic"

    def __lt__(self, other):
        return NotImplemented


@pytest.fixture
def generic_backend():
    return GenericComputationBackend()


class TestComputationBackend:
    def test_eq(self, generic_backend):
        assert generic_backend == generic_backend
        assert generic_backend == generic_backend.local_specifier
        assert generic_backend != 0

    def test_hash_smoke(self, generic_backend):
        assert isinstance(hash(generic_backend), int)

    def test_repr_smoke(self, generic_backend):
        assert isinstance(repr(generic_backend), str)

    def test_from_str_cpu(self):
        string = "cpu"
        backend = cb.ComputationBackend.from_str(string)
        assert isinstance(backend, cb.CPUBackend)

    @pytest.mark.parametrize(
        ("major", "minor", "string"),
        [
            pytest.param(major, minor, string, id=string)
            for major, minor, string in (
                (12, 3, "cu123"),
                (12, 3, "cu12.3"),
                (12, 3, "cuda123"),
                (12, 3, "cuda12.3"),
            )
        ],
    )
    def test_from_str_cuda(self, major, minor, string):
        backend = cb.ComputationBackend.from_str(string)
        assert isinstance(backend, cb.CUDABackend)
        assert backend.major == major
        assert backend.minor == minor

    @pytest.mark.parametrize(
        ("major", "minor", "patch", "string"),
        [
            pytest.param(major, minor, patch, string, id=string)
            for major, minor, patch, string in (
                (4, 5, 2, "rocm4.5.2"),
                (5, 0, None, "rocm5.0"),
            )
        ],
    )
    def test_from_str_rocm(self, major, minor, patch, string):
        backend = cb.ComputationBackend.from_str(string)
        assert isinstance(backend, cb.ROCmBackend)
        assert backend.major == major
        assert backend.minor == minor
        assert backend.patch == patch

    @pytest.mark.parametrize("string", (("unknown", "cudnn")))
    def test_from_str_unknown(self, string):
        with pytest.raises(ValueError, match=string):
            cb.ComputationBackend.from_str(string)
    def test_from_str_vulkan(self, major, minor, patch, string):
        backend = cb.ComputationBackend.from_str(string)
        assert isinstance(backend, cb.VulkanBackend)
        assert backend.major == major
        assert backend.minor == minor
        assert backend.patch == patch

    @pytest.mark.parametrize("string", (("unknown", "vulkan")))
    def test_from_str_unknown(self, string):
        with pytest.raises(ValueError, match=string):
            cb.ComputationBackend.from_str(string)


class TestCPUBackend:
    def test_eq(self):
        backend = cb.CPUBackend()
        assert backend == "cpu"


class TestCUDABackend:
    def test_eq(self):
        major = 42
        minor = 21
        backend = cb.CUDABackend(major, minor)
        assert backend == f"cu{major}{minor}"


class TestROCmBackend:
    @pytest.mark.parametrize("patch", [10, None])
    def test_eq_with_patch(self, patch):
        major = 42
        minor = 21
        backend = cb.ROCmBackend(major, minor, patch)
        assert (
            backend == f"rocm{major}.{minor}{f'.{patch}' if patch is not None else ''}"
        )

class TestVulkanBackend:
    @pytest.mark.parametrize("patch", [10, None])
    def test_eq_with_patch(self, patch):
        major = 42
        minor = 21
        backend = cb.VulkanBackend(major, minor, patch)
        assert (
            backend == f"vulkan{major}.{minor}{f'.{patch}' if patch is not None else ''}"
        )


class TestOrdering:
    def test_cpu(self):
        assert cb.CPUBackend() < cb.CUDABackend(0, 0)
        assert cb.VulkanBackend() < cb.CUDABackend(0, 0)
        assert cb.CPUBackend() < cb.VulkanBackend(0, 0)

    def test_cuda(self):
        assert cb.CUDABackend(0, 0) > cb.CPUBackend()
        assert cb.CUDABackend(1, 2) < cb.CUDABackend(2, 1)
        assert cb.CUDABackend(2, 1) < cb.CUDABackend(10, 0)

    def test_rocm(self):
        assert cb.ROCmBackend(0, 0, 0) > cb.CPUBackend()
        assert cb.ROCmBackend(0, 0, 0) > cb.VulkanBackend()

        assert cb.ROCmBackend(1, 2, 3) < cb.ROCmBackend(3, 2, 1)
        assert cb.ROCmBackend(3, 2, 1) < cb.ROCmBackend(10, 9, 8)

        assert cb.ROCmBackend(1, 2) < cb.ROCmBackend(1, 2, 0)
        assert cb.ROCmBackend(1, 2, 0) > cb.ROCmBackend(1, 2)

    def test_cuda_vs_rocm(self):
        cuda_backend = cb.CUDABackend(1, 2)
        rocm_backend = cb.ROCmBackend(1, 2)

        with pytest.raises(TypeError):
            cuda_backend < rocm_backend

        with pytest.raises(TypeError):
            rocm_backend < cuda_backend


@pytest.fixture
def patch_nvidia_driver_version(mocker):
    def factory(version):
        return mocker.patch(
            "light_the_torch._cb.subprocess.run",
            return_value=SimpleNamespace(stdout=f"driver_version\n{version}"),
        )

    return factory


def cuda_backends_params():
    params = []
    for system, minimum_driver_versions in cb._MINIMUM_DRIVER_VERSIONS.items():
        cuda_versions, driver_versions = zip(*sorted(minimum_driver_versions.items()))
        cuda_backends = tuple(
            cb.CUDABackend(version.major, version.minor) for version in cuda_versions
        )

        # latest driver supports every backend
        params.append(
            pytest.param(
                system,
                str(driver_versions[-1]),
                set(cuda_backends),
                id=f"{system.lower()}-latest",
            )
        )

        # outdated driver supports no backend
        params.append(
            pytest.param(
                system,
                str(driver_versions[0].major - 1),
                {},
                id=f"{system.lower()}-outdated",
            )
        )

    return pytest.mark.parametrize(
        ("system", "nvidia_driver_version", "compatible_cuda_backends"), params
    )


class TestDetectCompatibleComputationBackends:
    def test_no_nvidia_driver(self, mocker):
        mocker.patch(
            "light_the_torch._cb.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ""),
        )

        assert cb.detect_compatible_computation_backends() == {cb.CPUBackend()}

    @cuda_backends_params()
    def test_cuda_backends(
        self,
        mocker,
        patch_nvidia_driver_version,
        system,
        nvidia_driver_version,
        compatible_cuda_backends,
    ):
        mocker.patch("light_the_torch._cb.platform.system", return_value=system)
        patch_nvidia_driver_version(nvidia_driver_version)

        backends = cb.detect_compatible_computation_backends()
        assert backends == {cb.CPUBackend(), *compatible_cuda_backends}

    @skip_if_nvidia_driver_unavailable
    def test_cuda_backend(self):
        backend_types = {
            type(backend) for backend in cb.detect_compatible_computation_backends()
        }
        assert cb.CUDABackend in backend_types
