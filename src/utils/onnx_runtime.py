from __future__ import annotations

import logging
import ctypes
from pathlib import Path

log = logging.getLogger(__name__)


def prepare_onnx_runtime(device: str) -> None:
    """Preload CUDA/cuDNN from NVIDIA pip wheels before creating ONNX sessions."""
    if not device.startswith("cuda"):
        return
    import torch  # Loads CUDA libraries bundled with the matching PyTorch wheel.

    try:
        package_root = Path(torch.__file__).resolve().parents[1] / "nvidia"
        patterns = [
            "libcudart.so.11.0", "libnvrtc.so.11.2", "libcublas.so.11", "libcublasLt.so.11",
            "libcufft.so.10", "libcurand.so.10", "libcusolver.so.11", "libcusparse.so.11",
            "libnvToolsExt.so.1", "libnccl.so.2", "libcudnn*.so.8",
        ]
        for pattern in patterns:
            for library in package_root.glob(f"**/lib/{pattern}"):
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    except Exception as exc:
        raise RuntimeError(f"Could not preload CUDA/cuDNN libraries required by ONNX Runtime: {exc}") from exc

    import onnxruntime as ort

    # ORT 1.20's CUDA 11.8 build uses the CUDA/cuDNN libraries preloaded by PyTorch.
    # Newer CUDA 12 builds expose preload_dlls(), but this project pins the older pair
    # to keep Pascal GPUs supported on the detected host.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime does not expose CUDAExecutionProvider.")
    log.info("Loaded matching PyTorch and ONNX Runtime CUDA/cuDNN libraries")


def make_session_options():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options
