from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from src.utils.checksums import sha256_file
from src.utils.config import ROOT

log = logging.getLogger(__name__)


def verify_model_artifact(path: str | Path) -> None:
    """Refuse a changed downloaded weight when a manifest checksum is available."""
    target = Path(os.path.abspath(path))
    manifest_path = ROOT / "models" / "manifest.yaml"
    if not manifest_path.is_file() or not target.is_file():
        return
    try:
        relative = target.relative_to(ROOT).as_posix()
    except ValueError:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        expected_hashes = {
            row.get("sha256") for row in manifest.get("weight_files", [])
            if Path(str(row.get("path", ""))).name == target.name and row.get("sha256")
        }
        if not expected_hashes:
            log.warning("Model weight is outside this project and has no recorded checksum: %s", target.name)
            return
        if sha256_file(target) not in expected_hashes:
            raise RuntimeError(f"Model artifact checksum mismatch for external file {target.name}; refusing to load unpinned weights.")
        return
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    expected = next((row.get("sha256") for row in manifest.get("weight_files", []) if row.get("path") == relative), None)
    if not expected:
        expected = next((asset["sha256"] for asset in manifest.get("release_assets", {}).values()
                         if asset["path"] == relative), None)
    if not expected:
        for model in manifest.get("models", {}).values():
            directory = model.get("directory")
            if directory:
                expected = next((item["sha256"] for item in model.get("files", [])
                                 if f"{directory}/{item['name']}" == relative), None)
                if expected:
                    break
    if not expected:
        log.warning("No model checksum recorded for %s", relative)
        return
    actual = sha256_file(target)
    if actual != expected:
        raise RuntimeError(f"Model artifact checksum mismatch for {relative}; refusing to load unpinned weights.")


def release_artifact(key: str) -> Path:
    """Verify published release assets before exposing them to a model loader."""
    import shutil
    from tempfile import NamedTemporaryFile
    from urllib.request import urlopen

    catalog = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    entry = catalog["artifacts"][key]
    target = ROOT / entry["path"]
    if target.is_file():
        if sha256_file(target) != entry["sha256"]:
            raise RuntimeError(f"Model artifact checksum mismatch: {target.name}")
        return target
    if os.getenv("ANPR_OFFLINE") == "1":
        raise FileNotFoundError(f"Offline mode: missing {target.name}. Run the model download command online.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}-", delete=False) as temporary:
        partial = Path(temporary.name)
        try:
            with urlopen(entry["url"], timeout=45) as response:
                shutil.copyfileobj(response, temporary)
            temporary.flush()
            if sha256_file(partial) != entry["sha256"]:
                raise RuntimeError(f"Downloaded artifact checksum mismatch: {target.name}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    return target
