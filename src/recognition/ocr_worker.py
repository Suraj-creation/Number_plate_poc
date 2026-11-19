"""Bounded subprocess transport for local optional recognition engines."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import select
import subprocess
from time import monotonic
from weakref import finalize

import cv2


def _stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            stream.close()


class OCRWorkerError(RuntimeError):
    pass


class OCRWorker:
    def __init__(self, python, module, prefix, *, root, startup_timeout=90, request_timeout=30, extra_env=None):
        self.prefix = prefix.encode("ascii") + b" "
        self.timeout = request_timeout
        self.pending = bytearray()
        self.process = subprocess.Popen(
            [python, "-u", "-m", module], cwd=Path(root), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            bufsize=0, env={**os.environ, **(extra_env or {})},
        )
        self.close = finalize(self, _stop, self.process)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        try:
            self.receive(monotonic() + startup_timeout)
        except BaseException:
            self.close()
            raise

    def receive(self, deadline):
        stdout = self.process.stdout.fileno()
        while monotonic() < deadline:
            if b"\n" in self.pending:
                line, _, self.pending = self.pending.partition(b"\n")
                if line.startswith(self.prefix):
                    try:
                        value = json.loads(line[len(self.prefix):])
                    except (ValueError, UnicodeError) as exc:
                        raise OCRWorkerError("OCR worker returned invalid JSON.") from exc
                    if not isinstance(value, dict):
                        raise OCRWorkerError("OCR worker returned an invalid evidence object.")
                    return value
                continue
            if not select.select([stdout], [], [], max(0, deadline - monotonic()))[0]:
                break
            chunk = os.read(stdout, 65536)
            if not chunk:
                raise OCRWorkerError("OCR worker exited before returning evidence.")
            self.pending.extend(chunk)
            if len(self.pending) > 1024 * 1024:
                raise OCRWorkerError("OCR worker response was too large.")
        raise OCRWorkerError("OCR worker exceeded its response deadline.")

    def predict(self, crop):
        # Bound encoded input size before attempting to write to the worker.
        h, w = crop.shape[:2]
        if not h or not w:
            return {"raw": "", "raw_score": None, "minimum_char_score": None}
        if max(h, w) > 1920:
            scale = 1920 / max(h, w)
            crop = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", crop)
        if not ok or len(encoded) > 12 * 1024 * 1024:
            raise OCRWorkerError("Could not encode a bounded crop for the OCR worker.")
        request = memoryview(base64.b64encode(encoded) + b"\n")
        deadline = monotonic() + self.timeout
        try:
            stdin = self.process.stdin.fileno()
            while request:
                if not select.select([], [stdin], [], max(0, deadline - monotonic()))[1]:
                    raise OCRWorkerError("OCR worker exceeded its input deadline.")
                request = request[os.write(stdin, request):]
            return self.receive(deadline)
        except (OSError, OCRWorkerError) as exc:
            self.close()
            raise OCRWorkerError("Optional OCR worker failed or timed out. Check the local worker runtime.") from exc
