import platform
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Set

from pip._vendor.packaging.version import InvalidVersion, Version


class ComputationBackend(ABC):
    @property
    @abstractmethod
    def local_specifier(self) -> str:
        pass

    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        pass

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ComputationBackend):
            return self.local_specifier == other.local_specifier
        elif isinstance(other, str):
            return self.local_specifier == other
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.local_specifier)

    def __repr__(self) -> str:
        return self.local_specifier

    @classmethod
    def from_str(cls, string: str) -> "ComputationBackend":
        parse_error = ValueError(f"Unable to parse {string} into a computation backend")
        string = string.strip().lower()
        if string == "cpu":
            return CPUBackend()
        elif string.startswith("cu"):
            match = re.match(r"^cu(da)?(?P<version>[\d.]+)$", string)
            if match is None:
                raise parse_error

            version = match.group("version")
            if "." in version:
                major, minor = version.split(".")
            else:
                major = version[:-1]
                minor = version[-1]

            return CUDABackend(int(major), int(minor))
        elif string.startswith("vulkan") or string.startswith("vk"):
            # Accept: "vulkan", "vulkan1.2", "vk1.2", "vk1"
            match = re.match(r"^(?:vulkan|vk)(?P<version>[\d.]+)?$", string)
            if match is None:
                raise parse_error

            version = match.group("version")
            if not version:
                return VulkanBackend()
            if "." in version:
                parts = version.split(".")
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
            else:
                major = int(version)
                minor = 0
            return VulkanBackend(major, minor)
        elif string.startswith("rocm"):
            match = re.match(r"^rocm(?P<version>[\d.]+)$", string)
            if match is None:
                raise parse_error

            parts = match["version"].split(".")
            if len(parts) not in {2, 3}:
                raise parse_error

            return ROCmBackend(*[int(part) for part in parts])
        else:
            raise parse_error


class CPUBackend(ComputationBackend):
    @property
    def local_specifier(self) -> str:
        return "cpu"

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, ComputationBackend):
            return NotImplemented

        return True


class CUDABackend(ComputationBackend):
    def __init__(self, major: int, minor: int) -> None:
        self.major = major
        self.minor = minor

    @property
    def local_specifier(self) -> str:
        return f"cu{self.major}{self.minor}"

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, CPUBackend):
            return False
        elif isinstance(other, ROCmBackend):
            raise TypeError("Refusing to order a CUDA and a ROCm computation backend.")
        elif not isinstance(other, CUDABackend):
            return NotImplemented

        return (self.major, self.minor) < (other.major, other.minor)


class ROCmBackend(ComputationBackend):
    def __init__(self, major, minor, patch=None):
        self.major = major
        self.minor = minor
        self.patch = patch

    @property
    def local_specifier(self) -> str:
        parts = [self.major, self.minor]
        if self.patch is not None:
            parts.append(self.patch)
        return f"rocm{'.'.join(str(part) for part in parts)}"

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, CPUBackend):
            return False
        elif isinstance(other, CUDABackend):
            raise TypeError("Refusing to order a ROCm and a CUDA computation backend.")
        elif not isinstance(other, ROCmBackend):
            return NotImplemented

        if (self.major, self.minor) < (other.major, other.minor):
            return True
        elif (self.major, self.minor) > (other.major, other.minor):
            return False

        if self.patch is not None and other.patch is None:
            return False
        elif self.patch is None and other.patch is not None:
            return True
        else:
            return self.patch < other.patch


class VulkanBackend(ComputationBackend):
    def __init__(self, major: Optional[int] = None, minor: Optional[int] = None) -> None:
        # major/minor can be None when unspecified
        self.major = major
        self.minor = minor

    @property
    def local_specifier(self) -> str:
        if self.major is None:
            return "vulkan"
        if self.minor is None:
            return f"vulkan{self.major}"
        return f"vulkan{self.major}.{self.minor}"

    def __lt__(self, other: Any) -> bool:
        # Vulkan is a GPU backend; we treat it as greater than CPU but refuse to order with CUDA/ROCm
        if isinstance(other, CPUBackend):
            return False
        elif isinstance(other, (CUDABackend, ROCmBackend)):
            raise TypeError("Refusing to order a Vulkan and a CUDA/ROCm computation backend.")
        elif not isinstance(other, VulkanBackend):
            return NotImplemented

        # Compare versions when both have them defined. Missing parts are considered smaller.
        self_version = (self.major if self.major is not None else 0, self.minor if self.minor is not None else 0)
        other_version = (other.major if other.major is not None else 0, other.minor if other.minor is not None else 0)
        return self_version < other_version


def _detect_nvidia_driver_version() -> Optional[Version]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return Version(result.stdout.splitlines()[-1])
    except (FileNotFoundError, subprocess.CalledProcessError, InvalidVersion):
        return None


_MINIMUM_DRIVER_VERSIONS = {
    "Linux": {
        # Table 2 from
        # https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
        Version("13.1"): Version("580.65.06"),
        Version("13.0"): Version("580.65.06"),
        Version("12.9"): Version("525.60.13"),
        Version("12.8"): Version("525.60.13"),
        Version("12.6"): Version("525.60.13"),
        Version("12.5"): Version("525.60.13"),
        Version("12.4"): Version("525.60.13"),
        Version("12.3"): Version("525.60.13"),
        Version("12.2"): Version("525.60.13"),
        Version("12.1"): Version("525.60.13"),
        Version("12.0"): Version("525.60.13"),
        # Table 2 from
        # https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html
        Version("11.8"): Version("450.80.02"),
        Version("11.7"): Version("450.80.02"),
        Version("11.6"): Version("450.80.02"),
        Version("11.5"): Version("450.80.02"),
        Version("11.4"): Version("450.80.02"),
        Version("11.3"): Version("450.80.02"),
        Version("11.2"): Version("450.80.02"),
        Version("11.1"): Version("450.80.02"),
        Version("11.0"): Version("450.36.06"),
        # Table 1 from
        # https://docs.nvidia.com/cuda/archive/10.2/cuda-toolkit-release-notes/index.html
        Version("10.2"): Version("440.33"),
        Version("10.1"): Version("418.39"),
        Version("10.0"): Version("410.48"),
        Version("9.2"): Version("396.26"),
        Version("9.1"): Version("390.46"),
        Version("9.0"): Version("384.81"),
        Version("8.0"): Version("375.26"),
    },
    "Windows": {
        # Table 2 from
        # https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
        Version("13.1"): Version("580.0.0"),
        Version("13.0"): Version("580.0.0"),
        Version("12.9"): Version("528.33"),
        Version("12.8"): Version("528.33"),
        Version("12.6"): Version("528.33"),
        Version("12.5"): Version("528.33"),
        Version("12.4"): Version("528.33"),
        Version("12.3"): Version("528.33"),
        Version("12.2"): Version("528.33"),
        Version("12.1"): Version("528.33"),
        Version("12.0"): Version("528.33"),
        # Table 2 from
        # https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html
        Version("11.8"): Version("452.39"),
        Version("11.7"): Version("452.39"),
        Version("11.6"): Version("452.39"),
        Version("11.5"): Version("452.39"),
        Version("11.4"): Version("452.39"),
        Version("11.3"): Version("452.39"),
        Version("11.2"): Version("452.39"),
        Version("11.1"): Version("452.39"),
        Version("11.0"): Version("451.22"),
        # Table 1 from
        # https://docs.nvidia.com/cuda/archive/10.2/cuda-toolkit-release-notes/index.html
        Version("10.2"): Version("441.22"),
        Version("10.1"): Version("418.96"),
        Version("10.0"): Version("411.31"),
        Version("9.2"): Version("398.26"),
        Version("9.1"): Version("391.29"),
        Version("9.0"): Version("385.54"),
        Version("8.0"): Version("376.51"),
    },
}


def _detect_compatible_cuda_backends() -> List[CUDABackend]:
    driver_version = _detect_nvidia_driver_version()
    if not driver_version:
        return []

    minimum_driver_versions = _MINIMUM_DRIVER_VERSIONS.get(platform.system())
    if not minimum_driver_versions:
        return []

    return [
        CUDABackend(cuda_version.major, cuda_version.minor)
        for cuda_version, minimum_driver_version in minimum_driver_versions.items()
        if driver_version >= minimum_driver_version
    ]


def _detect_vulkan_backends() -> List[VulkanBackend]:
    """
    Detect Vulkan runtime using `vulkaninfo` if available and parse an instance version.
    Returns a list with a single VulkanBackend (with parsed major/minor if found) or empty list.
    """
    try:
        result = subprocess.run(
            ["vulkaninfo"],
            check=True,
            capture_output=True,
            text=True,
        )
        # Look for lines such as:
        # "Vulkan Instance Version: 1.2.162" or "Vulkan Instance Version: 1.3.204"
        for line in result.stdout.splitlines():
            m = re.search(r"Vulkan Instance Version:\s*(?P<version>[\d.]+)", line)
            if m:
                ver = m.group("version")
                parts = ver.split(".")
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                return [VulkanBackend(major, minor)]
        # If vulkaninfo ran but we couldn't parse a version, still assume Vulkan exists.
        return [VulkanBackend()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def detect_compatible_computation_backends() -> Set[ComputationBackend]:
    return {*_detect_compatible_cuda_backends(), *_detect_vulkan_backends(), CPUBackend()}