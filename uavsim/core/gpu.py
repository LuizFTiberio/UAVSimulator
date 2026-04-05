"""GPU detection, configuration, and status reporting.

Auto-detects available GPUs (AMD / NVIDIA / Intel) and configures JAX
to use the best available backend.  Users can override via:

* Environment variable ``UAVSIM_GPU`` (values: ``cpu``, ``cuda``, ``rocm``,
  ``metal``, ``auto``; default ``auto``).
* Calling :func:`configure_gpu` explicitly at startup.

Typical usage::

    from uavsim.core.gpu import gpu_info, configure_gpu

    # Auto-detect and configure (happens on import if UAVSIM_GPU != "cpu")
    info = configure_gpu()
    print(info)

    # Force CPU-only
    info = configure_gpu(backend="cpu")
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── enums ────────────────────────────────────────────────────────────────────

class GPUVendor(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"
    NONE = "none"


class JAXBackend(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    TPU = "tpu"


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    """Snapshot of detected GPU hardware and active JAX backend."""

    vendor: GPUVendor = GPUVendor.NONE
    device_name: str = ""
    jax_backend: JAXBackend = JAXBackend.CPU
    jax_devices: list[str] = field(default_factory=list)
    mjx_available: bool = False
    vram_mb: int = 0

    def __str__(self) -> str:
        lines = [
            f"GPU vendor   : {self.vendor.value}",
            f"Device       : {self.device_name or 'N/A'}",
            f"VRAM         : {self.vram_mb} MB" if self.vram_mb else "VRAM         : unknown",
            f"JAX backend  : {self.jax_backend.value}",
            f"JAX devices  : {', '.join(self.jax_devices) or 'N/A'}",
            f"MJX available: {self.mjx_available}",
        ]
        return "\n".join(lines)

    @property
    def has_gpu(self) -> bool:
        return self.jax_backend != JAXBackend.CPU

    def summary(self) -> str:
        if self.has_gpu:
            return (
                f"{self.vendor.value.upper()} {self.device_name} "
                f"via JAX/{self.jax_backend.value}"
            )
        return "CPU only"


# ── module-level state ───────────────────────────────────────────────────────

_gpu_info: Optional[GPUInfo] = None


def gpu_info() -> GPUInfo:
    """Return the current GPU info (auto-detects on first call)."""
    global _gpu_info
    if _gpu_info is None:
        _gpu_info = configure_gpu()
    return _gpu_info


# ── detection helpers ────────────────────────────────────────────────────────

def _detect_gpu_vendor_linux() -> tuple[GPUVendor, str]:
    """Detect GPU vendor on Linux via sysfs and lspci."""
    # PCI vendor IDs
    VENDOR_MAP = {
        "0x10de": GPUVendor.NVIDIA,
        "0x1002": GPUVendor.AMD,
        "0x8086": GPUVendor.INTEL,
    }

    # Try sysfs first (no subprocess needed)
    drm = Path("/sys/class/drm")
    if drm.exists():
        for card in sorted(drm.glob("card[0-9]*")):
            vendor_file = card / "device" / "vendor"
            if vendor_file.exists():
                vid = vendor_file.read_text().strip().lower()
                if vid in VENDOR_MAP:
                    # Try to get device name
                    name = _read_device_name_sysfs(card)
                    return VENDOR_MAP[vid], name

    # Fallback: lspci
    try:
        out = subprocess.run(
            ["lspci", "-nn"],
            capture_output=True, text=True, timeout=5,
        )
        for line in out.stdout.lower().splitlines():
            if "vga" in line or "3d controller" in line or "display" in line:
                if "nvidia" in line:
                    return GPUVendor.NVIDIA, _extract_lspci_name(line)
                if "amd" in line or "radeon" in line or "advanced micro" in line:
                    return GPUVendor.AMD, _extract_lspci_name(line)
                if "intel" in line:
                    return GPUVendor.INTEL, _extract_lspci_name(line)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return GPUVendor.UNKNOWN, ""


def _read_device_name_sysfs(card: Path) -> str:
    """Best-effort device name from sysfs or lspci."""
    # Try the uevent file for PCI slot, then lspci for the name
    uevent = card / "device" / "uevent"
    if uevent.exists():
        slot = ""
        for line in uevent.read_text().splitlines():
            if line.startswith("PCI_SLOT_NAME="):
                slot = line.split("=", 1)[1]
                break
        if slot:
            try:
                out = subprocess.run(
                    ["lspci", "-s", slot],
                    capture_output=True, text=True, timeout=5,
                )
                if out.stdout.strip():
                    return _extract_lspci_name(out.stdout.strip())
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
    return ""


def _extract_lspci_name(line: str) -> str:
    """Extract device name from a lspci output line."""
    # Typical: "06:00.0 VGA compatible controller: AMD ... [Radeon ...]"
    parts = line.split(":", 2)
    if len(parts) >= 3:
        return parts[2].strip()
    return line.strip()


def _detect_gpu_vendor_darwin() -> tuple[GPUVendor, str]:
    """Detect GPU on macOS."""
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        text = out.stdout.lower()
        if "apple" in text:
            # Extract chipset name
            for line in out.stdout.splitlines():
                if "chipset model" in line.lower() or "chip model" in line.lower():
                    return GPUVendor.APPLE, line.split(":", 1)[-1].strip()
            return GPUVendor.APPLE, "Apple Silicon"
        if "amd" in text or "radeon" in text:
            return GPUVendor.AMD, "AMD Radeon"
        if "nvidia" in text:
            return GPUVendor.NVIDIA, "NVIDIA"
        if "intel" in text:
            return GPUVendor.INTEL, "Intel"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return GPUVendor.UNKNOWN, ""


def detect_gpu_vendor() -> tuple[GPUVendor, str]:
    """Detect the primary GPU vendor and device name.

    Returns
    -------
    (vendor, device_name)
    """
    system = platform.system()
    if system == "Linux":
        return _detect_gpu_vendor_linux()
    if system == "Darwin":
        return _detect_gpu_vendor_darwin()
    # Windows / other — fall back to JAX device inspection
    return GPUVendor.UNKNOWN, ""


def _detect_vram_mb(vendor: GPUVendor) -> int:
    """Best-effort VRAM detection in MB."""
    try:
        if vendor == GPUVendor.NVIDIA:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return int(out.stdout.strip().split("\n")[0])
        if vendor == GPUVendor.AMD:
            # Try amdgpu sysfs
            for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
                vram_file = card / "device" / "mem_info_vram_total"
                if vram_file.exists():
                    return int(vram_file.read_text().strip()) // (1024 * 1024)
            # Fallback: rocm-smi
            out = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=5,
            )
            for line in out.stdout.splitlines():
                if "total" in line.lower():
                    parts = line.split()
                    for p in parts:
                        if p.isdigit():
                            val = int(p)
                            # rocm-smi may report bytes or MB
                            return val if val < 1_000_000 else val // (1024 * 1024)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0


# ── JAX backend detection + configuration ────────────────────────────────────

def _available_jax_backends() -> set[str]:
    """Return the set of JAX backend names that are actually installed."""
    import jax._src.xla_bridge as xb  # noqa: internal but stable across JAX 0.4+
    try:
        # JAX ≥ 0.4.20 exposes discover_pjrt_plugins or similar;
        # safest route is to check what factory functions are registered.
        return set(xb.backend_xla_client_factory().keys())
    except Exception:
        pass
    # Fallback: probe each commonly used platform
    available = {"cpu"}
    import jax
    for name in ("cuda", "rocm", "gpu", "tpu", "metal"):
        try:
            jax.devices(name)
            available.add(name)
        except (RuntimeError, ValueError):
            pass
    return available


def _get_jax_backend_and_devices() -> tuple[JAXBackend, list[str]]:
    """Query JAX for the active backend and device list."""
    import jax  # noqa: local import to avoid import-time side effects

    devices = jax.devices()
    device_strs = [f"{d.platform}:{d.id}" for d in devices]
    platform_name = devices[0].platform if devices else "cpu"

    backend_map = {
        "gpu": JAXBackend.CUDA,   # JAX reports CUDA as 'gpu'
        "cuda": JAXBackend.CUDA,
        "rocm": JAXBackend.ROCM,
        "tpu": JAXBackend.TPU,
        "metal": JAXBackend.METAL,
        "cpu": JAXBackend.CPU,
    }
    backend = backend_map.get(platform_name, JAXBackend.CPU)

    # Disambiguate CUDA vs ROCm when JAX reports 'gpu'
    if backend == JAXBackend.CUDA:
        try:
            if "rocm" in jax.lib.__version__.lower():
                backend = JAXBackend.ROCM
        except AttributeError:
            pass

    return backend, device_strs


def _check_mjx() -> bool:
    """Check whether MuJoCo MJX is importable."""
    try:
        import mujoco.mjx  # noqa: F401
        return True
    except ImportError:
        return False


# ── main entry point ─────────────────────────────────────────────────────────

def configure_gpu(backend: str | None = None) -> GPUInfo:
    """Detect hardware and configure JAX for the best backend.

    Parameters
    ----------
    backend : str, optional
        Force a specific backend: ``"cpu"``, ``"cuda"``, ``"rocm"``,
        ``"metal"``, ``"auto"`` (default).  Can also be set via the
        ``UAVSIM_GPU`` environment variable (the *backend* argument
        takes precedence).

    Returns
    -------
    GPUInfo
        Snapshot of the current GPU configuration.
    """
    global _gpu_info

    requested = (backend or os.environ.get("UAVSIM_GPU", "auto")).lower().strip()

    # If user explicitly wants CPU, configure JAX accordingly
    if requested == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax  # noqa
        info = GPUInfo(
            vendor=GPUVendor.NONE,
            jax_backend=JAXBackend.CPU,
            jax_devices=[f"{d.platform}:{d.id}" for d in jax.devices()],
        )
        _gpu_info = info
        logger.info("GPU disabled — using CPU only")
        return info

    # Detect hardware
    vendor, device_name = detect_gpu_vendor()
    vram = _detect_vram_mb(vendor)

    # Check which JAX backends are actually installed before hinting
    available = _available_jax_backends()

    if requested in ("cuda", "rocm", "metal"):
        jax_platform = "gpu" if requested in ("cuda", "rocm") else requested
        if jax_platform in available or requested in available:
            os.environ.setdefault("JAX_PLATFORMS", f"{jax_platform},cpu")
        else:
            logger.warning(
                "Requested JAX backend '%s' is not installed — falling back to CPU. "
                "Install jax[cuda12] or jax[rocm] for GPU support.",
                requested,
            )

    if requested == "auto":
        _preferred = {
            GPUVendor.AMD: "rocm",
            GPUVendor.NVIDIA: "cuda",
            GPUVendor.APPLE: "metal",
        }
        want = _preferred.get(vendor)
        if want and (want in available or "gpu" in available):
            jax_platform = "gpu" if want in ("cuda", "rocm") else want
            os.environ.setdefault("JAX_PLATFORMS", f"{jax_platform},cpu")
        # else: leave JAX_PLATFORMS unset → JAX picks the best it can

    # Now interrogate JAX
    jax_backend, jax_devices = _get_jax_backend_and_devices()

    info = GPUInfo(
        vendor=vendor,
        device_name=device_name,
        jax_backend=jax_backend,
        jax_devices=jax_devices,
        mjx_available=_check_mjx(),
        vram_mb=vram,
    )
    _gpu_info = info

    if info.has_gpu:
        logger.info("GPU configured: %s", info.summary())
    else:
        logger.info(
            "No GPU backend active (vendor=%s). "
            "Install jax[cuda12] or jax[rocm] for GPU acceleration.",
            vendor.value,
        )

    return info
