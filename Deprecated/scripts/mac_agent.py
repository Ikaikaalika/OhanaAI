#!/usr/bin/env python3
"""
OhanaAI Mac Training Agent (Polling)

This script runs on your Mac and polls your Vercel deployment for new
training data exports, trains the MLX model locally, then marks the data
as included in training.
"""

import argparse
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request, error
import subprocess
import shutil
from uuid import uuid4


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    with request.urlopen(url, timeout=timeout) as resp:
        dest.write_bytes(resp.read())


def _load_export(export_path: Path) -> Dict[str, Any]:
    with export_path.open("r") as fh:
        return json.load(fh)


def _write_training_batch(payload: Dict[str, Any], data_dir: Path) -> Path:
    data = payload.get("data")
    if data is None:
        data = payload.get("trainingData") or payload.get("training_data")

    if data is None:
        raise ValueError("Export payload missing training data array")

    out = {
        "metadata": payload.get("metadata", {}),
        "data": data,
    }

    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = data_dir / f"training_batch_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    return out_path


def _sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return safe.strip("._") or "source.ged"


def _download_source_gedcoms(payload: Dict[str, Any], data_dir: Path) -> list[Path]:
    source_files = payload.get("sourceFiles")
    if not isinstance(source_files, list):
        return []

    source_dir = data_dir / "source_gedcom"
    source_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for entry in source_files:
        if not isinstance(entry, dict):
            continue

        blob_url = entry.get("blobUrl")
        if not blob_url:
            continue

        gedcom_id = str(entry.get("gedcomFileId") or "unknown")
        original_name = str(entry.get("originalName") or "source.ged")
        safe_name = _sanitize_filename(original_name)
        destination = source_dir / f"{gedcom_id}_{safe_name}"

        try:
            _download(blob_url, destination, timeout=180)
            downloaded.append(destination)
        except Exception as e:
            print(f"[agent] failed to download GEDCOM source {gedcom_id}: {e}")

    return downloaded


def _run_training(cmd: list[str], cwd: Path) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def _run_command(cmd: list[str], cwd: Optional[Path] = None) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


def _post_multipart(url: str, fields: Dict[str, str], file_field: str, file_path: Path, content_type: str) -> Dict[str, Any]:
    boundary = f"----OhanaBoundary{uuid4().hex}"
    lines: list[bytes] = []

    for key, value in fields.items():
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        lines.append(str(value).encode())
        lines.append(b"\r\n")

    filename = file_path.name
    file_data = file_path.read_bytes()
    lines.append(f"--{boundary}\r\n".encode())
    lines.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode()
    )
    lines.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    lines.append(file_data)
    lines.append(b"\r\n")

    lines.append(f"--{boundary}--\r\n".encode())
    body = b"".join(lines)

    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with request.urlopen(req, timeout=120) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload) if payload else {}


def _convert_to_tfjs(
    onnx_path: Path,
    saved_model_dir: Path,
    tfjs_dir: Path,
    onnx_to_tf_cmd: Optional[str],
    tfjs_converter_cmd: Optional[str],
) -> bool:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    if tfjs_dir.exists():
        shutil.rmtree(tfjs_dir)

    saved_model_dir.mkdir(parents=True, exist_ok=True)
    tfjs_dir.mkdir(parents=True, exist_ok=True)

    if onnx_to_tf_cmd:
        cmd = shlex.split(onnx_to_tf_cmd.format(onnx=onnx_path, saved_model=saved_model_dir))
    else:
        cmd = ["onnx-tf", "convert", "-i", str(onnx_path), "-o", str(saved_model_dir)]

    if _run_command(cmd) != 0:
        raise RuntimeError("ONNX → TensorFlow conversion failed")

    if tfjs_converter_cmd:
        cmd = shlex.split(tfjs_converter_cmd.format(saved_model=saved_model_dir, tfjs_dir=tfjs_dir))
    else:
        cmd = [
            "tensorflowjs_converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            str(saved_model_dir),
            str(tfjs_dir),
        ]

    if _run_command(cmd) != 0:
        raise RuntimeError("TensorFlow → TF.js conversion failed")

    model_json = tfjs_dir / "model.json"
    if not model_json.exists():
        raise RuntimeError("TF.js model.json not found after conversion")

    return True


def _upload_tfjs(
    base_url: str,
    api_key: str,
    model_version: str,
    tfjs_dir: Path,
) -> Dict[str, Any]:
    upload_url = f"{base_url.rstrip('/')}/api/ml/upload-tfjs"
    model_json_url: Optional[str] = None
    uploaded_files: list[str] = []

    for file_path in sorted(tfjs_dir.glob("**/*")):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(tfjs_dir).as_posix()
        content_type = "application/json" if file_path.name.endswith(".json") else "application/octet-stream"
        response = _post_multipart(
            upload_url,
            {"apiKey": api_key, "modelVersion": model_version, "path": rel_path},
            "file",
            file_path,
            content_type,
        )
        uploaded_files.append(rel_path)
        if file_path.name == "model.json":
            model_json_url = response.get("url")

    return {
        "modelJsonUrl": model_json_url,
        "uploadedFiles": uploaded_files,
    }


def poll_once(
    base_url: str,
    api_key: str,
    data_dir: Path,
    output_dir: Path,
    training_script: Path,
    training_args: list[str],
    tfjs_enabled: bool,
    onnx_path: Optional[Path],
    tfjs_dir: Optional[Path],
    saved_model_dir: Optional[Path],
    onnx_to_tf_cmd: Optional[str],
    tfjs_converter_cmd: Optional[str],
) -> bool:
    export_url = f"{base_url.rstrip('/')}/api/ml/export-user-data"
    status_url = f"{base_url.rstrip('/')}/api/ml/training-status"
    mark_url = f"{base_url.rstrip('/')}/api/ml/mark-trained"
    source_gedcom_files: list[Path] = []

    try:
        response = _post_json(export_url, {"apiKey": api_key, "includeMetadata": True})
    except error.HTTPError as e:
        print(f"[agent] export request failed: {e}")
        return False
    except Exception as e:
        print(f"[agent] export request error: {e}")
        return False

    if response.get("count", 0) == 0:
        print("[agent] no new training data")
        return False

    export_path = response.get("exportPath")
    export_blob_url = response.get("exportUrl")
    export_blob_path = response.get("blobPath")
    exported_ids = response.get("exportedIds", [])

    if not export_path and not export_blob_url:
        print("[agent] no exportPath or exportUrl provided")
        return False

    tmp_dir = data_dir / "_exports"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_export = tmp_dir / f"{response.get('filename', 'export')}"

    try:
        if export_blob_url:
            print(f"[agent] downloading export from blob: {export_blob_url}")
            _download(export_blob_url, local_export)
        else:
            local_export = Path(export_path)
    except Exception as e:
        print(f"[agent] failed to download export: {e}")
        return False

    try:
        payload = _load_export(local_export)
        training_batch = _write_training_batch(payload, data_dir)
        source_gedcom_files = _download_source_gedcoms(payload, data_dir)
        if source_gedcom_files:
            print(f"[agent] downloaded {len(source_gedcom_files)} GEDCOM source file(s)")
    except Exception as e:
        print(f"[agent] failed to prepare training batch: {e}")
        return False

    lock_path = data_dir / ".training.lock"
    if lock_path.exists():
        print("[agent] training already in progress; skipping this cycle")
        return False
    lock_path.write_text(datetime.now(timezone.utc).isoformat())

    # Notify training start
    try:
        _post_json(status_url, {
            "apiKey": api_key,
            "status": "training",
            "message": "Training started",
            "details": {
                "batchFile": str(training_batch),
                "count": response.get("count", 0),
                "sourceGedcomFiles": len(source_gedcom_files)
            }
        })
    except Exception:
        pass

    # Run training
    cmd = [sys.executable, str(training_script), "--data-dir", str(data_dir), "--output-dir", str(output_dir)]
    cmd.extend(training_args)
    print(f"[agent] running training: {' '.join(cmd)}")
    exit_code = _run_training(cmd, cwd=training_script.parent)
    try:
        lock_path.unlink()
    except Exception:
        pass

    if exit_code != 0:
        try:
            _post_json(status_url, {
                "apiKey": api_key,
                "status": "error",
                "message": f"Training failed with exit code {exit_code}"
            })
        except Exception:
            pass
        print(f"[agent] training failed with exit code {exit_code}")
        return False

    model_version = f"mlx-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    tfjs_model_url = None
    uploaded_files: list[str] = []

    if tfjs_enabled:
        try:
            resolved_onnx = onnx_path or (output_dir / "model.onnx")
            resolved_tfjs_dir = tfjs_dir or (output_dir / "tfjs" / model_version)
            resolved_saved_model = saved_model_dir or (output_dir / "_saved_model" / model_version)

            print(f"[agent] converting ONNX to TF.js (modelVersion={model_version})")
            _convert_to_tfjs(
                resolved_onnx,
                resolved_saved_model,
                resolved_tfjs_dir,
                onnx_to_tf_cmd,
                tfjs_converter_cmd,
            )

            print(f"[agent] uploading TF.js artifacts ({resolved_tfjs_dir})")
            upload_result = _upload_tfjs(base_url, api_key, model_version, resolved_tfjs_dir)
            tfjs_model_url = upload_result.get("modelJsonUrl")
            uploaded_files = upload_result.get("uploadedFiles", [])
        except Exception as e:
            try:
                _post_json(status_url, {
                    "apiKey": api_key,
                    "status": "error",
                    "message": f"TF.js export failed: {e}"
                })
            except Exception:
                pass
            print(f"[agent] TF.js export failed: {e}")
            return False

    # Notify success + mark trained
    try:
        _post_json(status_url, {
            "apiKey": api_key,
            "status": "ready",
            "message": "Training complete",
            "modelVersion": model_version,
            "details": {
                "sourceGedcomFiles": len(source_gedcom_files),
                "tfjsModelUrl": tfjs_model_url if tfjs_enabled else None,
                "tfjsFiles": uploaded_files if tfjs_enabled else []
            }
        })
    except Exception:
        pass

    try:
        _post_json(mark_url, {
            "apiKey": api_key,
            "exportedIds": exported_ids,
            "exportBlobPath": export_blob_path,
            "exportUrl": export_blob_url
        })
    except Exception as e:
        print(f"[agent] mark-trained failed: {e}")

    print("[agent] training complete")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="OhanaAI Mac training agent")
    parser.add_argument("--base-url", default=os.getenv("OHANA_API_BASE"), help="Vercel app base URL")
    parser.add_argument("--api-key", default=os.getenv("ML_EXPORT_API_KEY"), help="ML export API key")
    parser.add_argument("--data-dir", default=None, help="Training data directory")
    parser.add_argument("--output-dir", default=None, help="Model output directory")
    parser.add_argument("--training-script", default=None, help="Training script path")
    parser.add_argument("--training-args", default="", help="Extra training args")
    parser.add_argument("--skip-tfjs", action="store_true", help="Skip TF.js conversion/upload")
    parser.add_argument("--onnx-path", default=None, help="Override ONNX model path")
    parser.add_argument("--tfjs-dir", default=None, help="Override TF.js output directory")
    parser.add_argument("--saved-model-dir", default=None, help="Override TensorFlow SavedModel directory")
    parser.add_argument("--interval", type=int, default=300, help="Polling interval seconds")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    if not args.base_url:
        print("Missing --base-url or OHANA_API_BASE")
        return 1
    if not args.api_key:
        print("Missing --api-key or ML_EXPORT_API_KEY")
        return 1

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "training_data")
    output_dir = Path(args.output_dir) if args.output_dir else (repo_root / "models" / "parent_predictor")
    training_script = Path(args.training_script) if args.training_script else (script_dir / "training" / "train_model_mlx.py")
    training_args = [arg for arg in args.training_args.split(" ") if arg] if args.training_args else []
    tfjs_enabled = not args.skip_tfjs
    onnx_path = Path(args.onnx_path) if args.onnx_path else None
    tfjs_dir = Path(args.tfjs_dir) if args.tfjs_dir else None
    saved_model_dir = Path(args.saved_model_dir) if args.saved_model_dir else None
    onnx_to_tf_cmd = os.getenv("ONNX_TO_TF_CMD")
    tfjs_converter_cmd = os.getenv("TFJS_CONVERTER_CMD")

    if not training_script.exists():
        print(f"Training script not found: {training_script}")
        return 1

    print(f"[agent] base URL: {args.base_url}")
    print(f"[agent] data dir: {data_dir}")
    print(f"[agent] output dir: {output_dir}")
    print(f"[agent] training script: {training_script}")
    print(f"[agent] TF.js export: {'enabled' if tfjs_enabled else 'disabled'}")

    while True:
        poll_once(
            args.base_url,
            args.api_key,
            data_dir,
            output_dir,
            training_script,
            training_args,
            tfjs_enabled,
            onnx_path,
            tfjs_dir,
            saved_model_dir,
            onnx_to_tf_cmd,
            tfjs_converter_cmd,
        )
        if args.once:
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
