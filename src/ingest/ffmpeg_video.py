"""System FFmpeg decoder for local AV1 video on hosts whose OpenCV lacks a decoder."""

from fractions import Fraction
import json
from pathlib import Path
import subprocess

import numpy as np


class Av1Capture:
    def __init__(self, source, threads, max_pixels):
        path = str(Path(source).resolve())
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", path],
                capture_output=True, text=True, check=True, timeout=10)
            stream = json.loads(probe.stdout)["streams"][0]
            self.width, self.height = int(stream["width"]), int(stream["height"])
            self.fps = float(Fraction(stream["avg_frame_rate"]))
            self.frame_count = int(stream["nb_frames"]) if stream.get("nb_frames", "N/A").isdecimal() else 0
        except (OSError, subprocess.SubprocessError, KeyError, IndexError, ValueError, ZeroDivisionError) as exc:
            raise RuntimeError("Could not probe the AV1 file using system FFmpeg.") from exc
        if self.fps <= 0 or self.width <= 0 or self.height <= 0 or self.width * self.height > max_pixels:
            raise ValueError("Invalid AV1 frame rate or frame size above the configured pixel limit.")
        self.frame_bytes = self.width * self.height * 3
        self.failed = False
        try:
            self.process = subprocess.Popen(
                ["ffmpeg", "-nostdin", "-v", "error", "-threads", str(threads), "-c:v", "libdav1d",
                 "-noautorotate", "-i", path, "-map", "0:v:0", "-an", "-sn", "-threads", str(threads),
                 "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as exc:
            raise RuntimeError("Install FFmpeg with libdav1d to decode local AV1 videos.") from exc

    def read(self):
        data = self.process.stdout.read(self.frame_bytes)
        if len(data) != self.frame_bytes:
            code = self.process.wait(timeout=5)
            self.failed = bool(code or data)
            return False, None
        return True, np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)

    def release(self):
        self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
