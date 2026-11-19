from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class CudaUnavailable(RuntimeError):
    """The requested CUDA execution environment failed initialization."""


def device_report(probe: bool = True) -> dict:
    report = {"torch": None, "cuda_available": False, "devices": []}
    try:
        import torch

        report["torch"] = str(torch.__version__)
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_runtime"] = torch.version.cuda
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            report["devices"].append({"index": i, "name": props.name, "memory_gib": round(props.total_memory / 2**30, 2), "capability": f"{props.major}.{props.minor}"})
        if probe:
            for device in report["devices"]:
                try:
                    require_cuda(f"cuda:{device['index']}")
                    device["probe_ok"] = True
                except RuntimeError as exc:
                    device["probe_ok"] = False
                    device["probe_error"] = str(exc)
            report["torch_cuda_probe_ok"] = bool(report["devices"]) and all(d["probe_ok"] for d in report["devices"])
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import onnxruntime as ort

        report["onnxruntime"] = ort.__version__
        report["onnx_providers"] = ort.get_available_providers()
        report["onnx_cuda_preload_ok"] = None
        report["onnx_cuda_preload_method"] = "not_applicable" if not hasattr(ort, "preload_dlls") else "onnxruntime.preload_dlls"
        if "CUDAExecutionProvider" in report["onnx_providers"] and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(directory="")
                report["onnx_cuda_preload_ok"] = True
            except Exception as exc:
                report["onnx_cuda_preload_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        report["onnx_error"] = f"{type(exc).__name__}: {exc}"
    return report


def require_cuda(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise CudaUnavailable("CUDA was requested but torch.cuda.is_available() is false. See `python -m src.cli doctor`.")
    device_index = int(device.split(":", 1)[1]) if ":" in device else 0
    if device_index >= torch.cuda.device_count():
        raise CudaUnavailable(f"CUDA device {device_index} is unavailable. See `python -m src.cli doctor`.")
    try:
        probe = torch.nn.Conv2d(1, 1, 3, device=f"cuda:{device_index}")
        probe(torch.ones((1, 1, 8, 8), device=f"cuda:{device_index}"))
        torch.cuda.synchronize(device_index)
    except Exception as exc:
        raise CudaUnavailable(f"CUDA device {device_index} cannot run PyTorch inference: {exc}") from exc
