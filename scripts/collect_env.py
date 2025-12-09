#!/usr/bin/env python

import itertools
import platform
import subprocess

import importlib_metadata
import pip
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import InvalidVersion, Version


try:
    import light_the_torch
except ModuleNotFoundError:
    light_the_torch = None

NOT_AVAILABLE = "N/A"

# TODO: somehow merge this with light_the_torch._patch.PYTORCH_DISTRIBUTIONS to avoid
#  duplication
PYTORCH_DISTRIBUTIONS = {
    "torch",
    "torch_model_archiver",
    "torch_tb_profiler",
    "torcharrow",
    "torchaudio",
    "torchcsprng",
    "torchdata",
    "torchdistx",
    "torchserve",
    "torchtext",
    "torchvision",
}


def main():
    header("System")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python version: {platform.python_version()}")
    nvidia_driver_version = detect_nvidia_driver_version()
    print(f"NVIDIA driver version: {nvidia_driver_version or NOT_AVAILABLE}")

    header("Environment")
    for name, version in itertools.chain(
        [
            ("pip", pip.__version__),
            (
                "light_the_torch",
                light_the_torch.__version__ if light_the_torch else NOT_AVAILABLE,
            ),
        ],
        detect_pytorch_or_dependent_packages(),
    ):
        print(f"- `{name}=={version}`")

    # Vulkan capabilities / availability checks
    header("Vulkan")
    detect_vulkan_capabilities()


# TODO: somehow merge this with light_the_torch._cb._detect_nvidia_driver_version to
#  avoid duplication
def detect_nvidia_driver_version():
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


def detect_pytorch_or_dependent_packages():
    packages = {
        (dist.name, dist.version)
        for dist in importlib_metadata.distributions()
        if any(
            name in PYTORCH_DISTRIBUTIONS
            for name in itertools.chain(
                [dist.name],
                (
                    [Requirement(req_str).name for req_str in dist.requires]
                    if dist.requires
                    else []
                ),
            )
        )
    }
    return sorted(
        packages,
        key=lambda package: (package[0] not in PYTORCH_DISTRIBUTIONS, package[0]),
    )


def detect_vulkan_capabilities():
    """
    Print information about Vulkan capabilities relevant to PyTorch usage.

    This function checks:
    - whether PyTorch is importable
    - whether PyTorch was built/installed with Vulkan backend available
    - attempts to run `vulkaninfo` (if present) and prints a short snippet of its output

    For more context about PyTorch's Vulkan support and workflow see:
    https://docs.pytorch.org/tutorials/unstable/vulkan_workflow.html
    """
    try:
        import torch
    except Exception:
        print("- PyTorch: N/A (cannot check Vulkan without PyTorch)")
        print("- Vulkan backend available in PyTorch: N/A")
        print("- See: https://docs.pytorch.org/tutorials/unstable/vulkan_workflow.html")
        return

    # PyTorch basic info
    try:
        torch_version = torch.__version__
    except Exception:
        torch_version = NOT_AVAILABLE
    print(f"- PyTorch version: {torch_version}")

    # Check whether the Vulkan backend is available in this PyTorch build.
    has_vulkan = False
    try:
        backend_vulkan = getattr(torch.backends, "vulkan", None)
        if backend_vulkan is not None and hasattr(backend_vulkan, "is_available"):
            has_vulkan = bool(backend_vulkan.is_available())
    except Exception:
        has_vulkan = False

    print(f"- Vulkan backend available in PyTorch: {has_vulkan}")

    if not has_vulkan:
        # Provide guidance from the Vulkan workflow tutorial
        print(
            "- Note: PyTorch's Vulkan backend enables running tensor computations on "
            "Vulkan-capable GPUs (primarily targeted at mobile/embedded GPUs). "
            "It is experimental and may require building PyTorch with Vulkan support or "
            "installing a wheel that includes Vulkan. See the tutorial for details:"
        )
        print("  https://docs.pytorch.org/tutorials/unstable/vulkan_workflow.html")
        return

    # If Vulkan is available, try to gather system Vulkan info via `vulkaninfo` (if installed).
    try:
        result = subprocess.run(
            ["vulkaninfo"],
            check=True,
            capture_output=True,
            text=True,
        )
        vulkan_output = result.stdout or result.stderr or ""
        lines = vulkan_output.splitlines()
        if not lines:
            print("- `vulkaninfo` ran but produced no output.")
        else:
            # Print a short snippet of vulkaninfo to show device/driver details
            print("- `vulkaninfo` summary (first 10 non-empty lines):")
            printed = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                print(f"  {line}")
                printed += 1
                if printed >= 10:
                    break
    except FileNotFoundError:
        print("- `vulkaninfo` not found on PATH (Vulkan SDK/tools not installed).")
        print("  Installing Vulkan SDK / tools or adding `vulkaninfo` to PATH can give more detailed device info.")
    except subprocess.CalledProcessError:
        print("- `vulkaninfo` failed to run or returned an error; device details may still be available via platform tools.")

    # Offer lightweight guidance from the tutorial about how Vulkan is used in PyTorch
    print(
        "- Quick notes from the Vulkan workflow tutorial:\n"
        "  - PyTorch's Vulkan backend allows execution of tensor ops on Vulkan-capable GPUs and is especially\n"
        "    useful for mobile/embedded inference and certain computer-vision workloads.\n"
        "  - The workflow typically involves tracing/compiling models to use Vulkan kernels, and may require\n"
        "    image-format-aware data preparation (see tutorial for image preprocessing instructions).\n"
        "  - Because the backend is experimental, APIs and capabilities may change; consult the tutorial and\n"
        "    the PyTorch release notes for the most up-to-date instructions."
    )


def header(name):
    print()
    print(f"#### {name}")
    print()


if __name__ == "__main__":
    main()