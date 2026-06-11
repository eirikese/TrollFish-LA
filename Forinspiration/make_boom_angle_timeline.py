"""
Create an interactive hull timeline from GoPro boom-angle predictions.

Input:
    - GoPro video
    - trained PilotNet/BoomAngleNetLite/SailAngleNet checkpoint

Output directory:
    - boom_angle_timeline.html
    - boom_angle_timeline.csv
    - boom_angle_timeline.json
    - Hull.stl copy for the HTML viewer

The HTML uses the same hull transform and boom convention as hull_pnp_tool.html:
    mast mount: [0.0, 0.55, 0.0]
    azimuth: atan2(z, x), in degrees
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from angledetection import pilotnet as pilot  # noqa: E402
from angledetection import sail_angle_net as sail  # noqa: E402
from angledetection.validate_angle_predictions import (  # noqa: E402
    build_pilot_model,
    build_sail_model,
    detect_checkpoint_type,
    load_json_roi,
    resolve_device,
    validate_roi,
)


DEFAULT_HULL_STL = ROOT_DIR / "Hull.stl"
DEFAULT_OUT_DIR = ROOT_DIR / "angledetection" / "boom_angle_timelines"
DEFAULT_ROI_CONFIG = ROOT_DIR / "angledetection" / "gopro_boom_training_dataset" / "roi_config.json"
DEFAULT_WEIGHTS = ROOT_DIR / "angledetection" / "gopro_boom_training_dataset" / "best_pilotnet_boom.pth"

MAST_MOUNT = (0.0, 0.55, 0.0)
DEFAULT_BOOM_LENGTH_M = 2.7


def sample_indices(frame_count: int, fps: float, args) -> list[int]:
    start = max(0, int(round(args.start_sec * fps)))
    end_sec = args.end_sec if args.end_sec is not None else (frame_count - 1) / fps
    end = min(frame_count - 1, int(round(end_sec * fps)))
    if end < start:
        raise ValueError("--end-sec is before --start-sec")

    if args.frames:
        return sorted({max(0, min(frame_count - 1, int(v))) for v in args.frames})

    if args.sample_fps and args.sample_fps > 0:
        step = max(1, int(round(fps / args.sample_fps)))
    else:
        step = max(1, int(round(args.every_sec * fps)))

    indices = list(range(start, end + 1, step))
    if args.max_samples is not None:
        indices = indices[: max(1, int(args.max_samples))]
    return indices


def read_frame(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def iter_sampled_frames(cap: cv2.VideoCapture, indices: list[int], read_mode: str):
    if read_mode == "seek":
        for frame_index in indices:
            yield int(frame_index), read_frame(cap, int(frame_index))
        return

    if read_mode != "sequential":
        raise ValueError("--read-mode must be sequential or seek")
    if not indices:
        return

    next_frame = int(indices[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
    for frame_index in indices:
        frame_index = int(frame_index)
        if frame_index < next_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            next_frame = frame_index
        while next_frame < frame_index:
            ok = cap.grab()
            if not ok:
                raise RuntimeError(f"Could not skip to frame {frame_index}")
            next_frame += 1
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_index}")
        next_frame = frame_index + 1
        yield frame_index, frame


def frame_cache_bytes(num_frames: int, width: int, height: int) -> int:
    return int(num_frames * width * height * 3)


def format_bytes(num_bytes: int) -> str:
    gb = num_bytes / 1024 ** 3
    if gb >= 1:
        return f"{gb:.2f} GB"
    return f"{num_bytes / 1024 ** 2:.0f} MB"


def load_frames_to_ram(cap: cv2.VideoCapture, indices: list[int], read_mode: str, width: int, height: int, max_gb: float):
    estimate = frame_cache_bytes(len(indices), width, height)
    budget = max(0.0, float(max_gb)) * 1024 ** 3
    if estimate > budget:
        raise MemoryError(
            f"--frame-cache ram would use about {format_bytes(estimate)}, above --frame-cache-max-gb "
            f"{max_gb:g}. Use --frame-cache none, reduce --sample-fps, or raise the budget."
        )
    print(f"Reading {len(indices)} sampled frames into RAM ({format_bytes(estimate)} estimate)...")
    return list(iter_sampled_frames(cap, indices, read_mode))


def chunks(items: list[Any], size: int):
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i:i + size]


def vector_from_angles(
    azimuth_deg: float,
    elevation_deg: float,
    length_m: float,
) -> tuple[float, float, float]:
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    horizontal = float(length_m) * math.cos(el)
    return (
        horizontal * math.cos(az),
        float(length_m) * math.sin(el),
        horizontal * math.sin(az),
    )


def preprocess_pilot_batch(
    frames_bgr: list[np.ndarray],
    roi: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    top, bottom, left, right = roi
    batch = np.empty((len(frames_bgr), 3, pilot.INPUT_SIZE[0], pilot.INPUT_SIZE[1]), dtype=np.float32)
    for i, frame in enumerate(frames_bgr):
        h, w = frame.shape[:2]
        t = max(0, min(h - 1, top))
        b = max(t + 1, min(h, bottom))
        l = max(0, min(w - 1, left))
        r = max(l + 1, min(w, right))
        crop = frame[t:b, l:r]
        crop = cv2.resize(crop, (pilot.INPUT_SIZE[1], pilot.INPUT_SIZE[0]), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
        crop = crop.astype(np.float32) / 127.5 - 1.0
        batch[i] = np.transpose(crop, (2, 0, 1))
    return torch.from_numpy(batch).to(device, non_blocking=True)


@torch.no_grad()
def predict_pilot_batch(
    model,
    frames_bgr: list[np.ndarray],
    roi: tuple[int, int, int, int],
    device: torch.device,
    angle_mean: float,
    angle_std: float,
) -> list[float]:
    x = preprocess_pilot_batch(frames_bgr, roi, device)
    preds = model(x).float().detach().cpu().numpy().reshape(-1)
    return (preds * angle_std + angle_mean).astype(float).tolist()


@torch.no_grad()
def predict_sail_batch(
    model,
    frames_bgr: list[np.ndarray],
    cfg: sail.PreprocessConfig,
    device: torch.device,
    map1=None,
    map2=None,
    calib_size: tuple[int, int] | None = None,
) -> list[float]:
    tensors = [
        sail.preprocess_bgr(frame, cfg, map1=map1, map2=map2, calib_size=calib_size)
        for frame in frames_bgr
    ]
    x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
    pred_unit = model(x)
    return sail.unit_to_angle_deg(pred_unit.float()).detach().cpu().numpy().reshape(-1).astype(float).tolist()


def load_predictor(args, device: torch.device):
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model_type = detect_checkpoint_type(checkpoint, args.model_type)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if model_type == "pilot":
        model, arch, angle_mean, angle_std, target_col, roi = build_pilot_model(checkpoint, device, args.arch)
        if args.roi_config:
            roi = load_json_roi(args.roi_config)
        if args.roi is not None:
            roi = validate_roi(tuple(args.roi))

        def predictor(frames_bgr: list[np.ndarray]) -> list[float]:
            return predict_pilot_batch(model, frames_bgr, roi, device, angle_mean, angle_std)

        return predictor, {
            "model_type": "pilot",
            "model_label": f"Pilot ({arch})",
            "target_col": target_col,
            "roi": list(roi),
        }

    model, cfg, target_col = build_sail_model(checkpoint, device)
    if args.roi_config:
        roi = load_json_roi(args.roi_config)
        cfg = sail.PreprocessConfig(
            roi=roi,
            input_size=cfg.input_size,
            undistort=False,
            calib_path=cfg.calib_path,
            allow_calib_rescale=cfg.allow_calib_rescale,
            use_edge=cfg.use_edge,
        )
    if args.roi is not None:
        roi = validate_roi(tuple(args.roi))
        cfg = sail.PreprocessConfig(
            roi=roi,
            input_size=cfg.input_size,
            undistort=False,
            calib_path=cfg.calib_path,
            allow_calib_rescale=cfg.allow_calib_rescale,
            use_edge=cfg.use_edge,
        )

    undistort_maps: dict[str, Any] = {"map1": None, "map2": None, "calib_size": None}

    def predictor(frames_bgr: list[np.ndarray]) -> list[float]:
        if cfg.undistort and undistort_maps["map1"] is None:
            first = frames_bgr[0]
            map1, map2, calib_size = sail.load_fisheye_undistort_maps(
                cfg.calib_path,
                target_size=(first.shape[1], first.shape[0]),
                allow_rescale=cfg.allow_calib_rescale,
            )
            undistort_maps["map1"] = map1
            undistort_maps["map2"] = map2
            undistort_maps["calib_size"] = calib_size
        return predict_sail_batch(
            model,
            frames_bgr,
            cfg,
            device,
            map1=undistort_maps["map1"],
            map2=undistort_maps["map2"],
            calib_size=undistort_maps["calib_size"],
        )

    return predictor, {
        "model_type": "sail",
        "model_label": "SailAngleNet",
        "target_col": target_col,
        "roi": list(cfg.roi),
    }


def predict_timeline(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = resolve_device(args.device)
    predictor, model_info = load_predictor(args, device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Bad video metadata: fps={fps}, frame_count={frame_count}")

    indices = sample_indices(frame_count, fps, args)
    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    print(f"Video: {args.video}")
    print(f"Frames: {frame_count}  FPS: {fps:.6f}  Size: {width}x{height}")
    print(f"Samples: {len(indices)}")
    print(f"Device: {device}")
    print(f"Model: {model_info['model_label']}")
    print(f"ROI: {model_info['roi']}")
    print(f"Batch size: {args.batch_size}")
    print(f"Read mode: {args.read_mode}")
    print(f"Frame cache: {args.frame_cache}")

    try:
        if args.frame_cache == "ram":
            frame_items = load_frames_to_ram(
                cap,
                indices,
                args.read_mode,
                width,
                height,
                args.frame_cache_max_gb,
            )
            batch_iter = chunks(frame_items, args.batch_size)
        else:
            def streaming_batches():
                batch = []
                for item in iter_sampled_frames(cap, indices, args.read_mode):
                    batch.append(item)
                    if len(batch) >= max(1, int(args.batch_size)):
                        yield batch
                        batch = []
                if batch:
                    yield batch

            batch_iter = streaming_batches()

        sample_index = 0
        for batch_items in batch_iter:
            frame_indices = [int(frame_index) for frame_index, _frame in batch_items]
            frames = [frame for _frame_index, frame in batch_items]
            raw_angles = predictor(frames)
            if len(raw_angles) != len(frames):
                raise RuntimeError(f"Predictor returned {len(raw_angles)} values for {len(frames)} frames")

            for frame_index, raw_angle in zip(frame_indices, raw_angles):
                raw_angle = float(raw_angle)
                display_angle = raw_angle * (-1.0 if args.invert_angle else 1.0) + float(args.angle_offset_deg)
                boom_vector = vector_from_angles(display_angle, args.elevation_deg, args.boom_length_m)
                boom_unit = vector_from_angles(display_angle, args.elevation_deg, 1.0)
                row = {
                    "sample_index": sample_index,
                    "frame_index": int(frame_index),
                    "timestamp_sec": float(frame_index / fps),
                    "raw_prediction_deg": raw_angle,
                    "boom_azimuth_deg": display_angle,
                    "boom_elevation_deg": float(args.elevation_deg),
                    "boom_length_m": float(args.boom_length_m),
                    "boom_vector_x_m": boom_vector[0],
                    "boom_vector_y_m": boom_vector[1],
                    "boom_vector_z_m": boom_vector[2],
                    "boom_unit_x": boom_unit[0],
                    "boom_unit_y": boom_unit[1],
                    "boom_unit_z": boom_unit[2],
                }
                rows.append(row)
                if (
                    (sample_index + 1) % max(1, int(args.progress_every)) == 0
                    or sample_index == 0
                    or sample_index == len(indices) - 1
                ):
                    print(
                        f"[{sample_index + 1:5d}/{len(indices):5d}] "
                        f"t={row['timestamp_sec']:8.2f}s  angle={display_angle:+8.2f} deg"
                    )
                sample_index += 1
    finally:
        cap.release()

    elapsed = max(1e-6, time.perf_counter() - start_time)
    print(f"Finished {len(rows)} predictions in {elapsed:.2f}s ({len(rows) / elapsed:.2f} samples/s)")

    metadata = {
        "video": str(Path(args.video).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "device": str(device),
        "video_fps": fps,
        "video_frame_count": frame_count,
        "video_width": width,
        "video_height": height,
        "model": model_info,
        "angle_offset_deg": float(args.angle_offset_deg),
        "invert_angle": bool(args.invert_angle),
        "boom_length_m": float(args.boom_length_m),
        "boom_elevation_deg": float(args.elevation_deg),
        "mast_mount_point_m": list(MAST_MOUNT),
        "batch_size": int(args.batch_size),
        "read_mode": args.read_mode,
        "frame_cache": args.frame_cache,
    }
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "samples": rows}, f, indent=2)


def html_template(data_obj: dict[str, Any], hull_base64: str) -> str:
    embedded_data = json.dumps(data_obj, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Boom Angle Hull Timeline</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: #121a22;
      --panel2: #18222c;
      --line: #2b3a47;
      --text: #f0f5f8;
      --muted: #a9b8c3;
      --accent: #65e0c2;
      --warn: #ffc76b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      overflow: hidden;
    }}
    .app {{
      height: 100vh;
      display: grid;
      grid-template-rows: 1fr 210px;
    }}
    .main {{
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      border-bottom: 1px solid var(--line);
    }}
    #scene {{ min-width: 0; min-height: 0; }}
    .side {{
      background: var(--panel);
      border-left: 1px solid var(--line);
      padding: 18px;
      display: grid;
      align-content: start;
      gap: 14px;
    }}
    h1 {{ margin: 0; font-size: 22px; }}
    .metric {{
      padding: 12px;
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 6px;
    }}
    .value {{ font-size: 18px; font-weight: 700; }}
    .controls {{
      padding: 14px 18px;
      background: #101820;
      display: grid;
      gap: 10px;
    }}
    .row {{
      display: grid;
      grid-template-columns: auto 1fr auto auto auto;
      gap: 10px;
      align-items: center;
    }}
    button {{
      border: 1px solid var(--line);
      background: var(--panel2);
      color: var(--text);
      border-radius: 8px;
      padding: 9px 12px;
      cursor: pointer;
      font-weight: 700;
    }}
    button:hover {{ border-color: var(--accent); }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    canvas#plot {{
      width: 100%;
      height: 110px;
      background: #0c1218;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .hint {{ color: var(--muted); font-size: 12px; line-height: 1.45; }}
    code {{ color: var(--accent); }}
  </style>
</head>
<body>
<div class="app">
  <div class="main">
    <div id="scene"></div>
    <aside class="side">
      <h1>Boom Angle Timeline</h1>
      <div class="metric"><div class="label">Time</div><div class="value" id="timeValue">--</div></div>
      <div class="metric"><div class="label">Boom azimuth</div><div class="value" id="angleValue">--</div></div>
      <div class="metric"><div class="label">Frame</div><div class="value" id="frameValue">--</div></div>
      <div class="metric"><div class="label">Model</div><div class="value" id="modelValue">--</div></div>
      <p class="hint">Azimuth follows the hull tool convention: <code>atan2(z, x)</code>. The boom is drawn from mast mount <code>[0, 0.55, 0]</code>.</p>
    </aside>
  </div>
  <div class="controls">
    <canvas id="plot"></canvas>
    <div class="row">
      <button id="playBtn">Play</button>
      <input id="scrub" type="range" min="0" max="0" step="1" value="0">
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
      <span class="hint" id="indexValue">0 / 0</span>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
<script>
const EMBEDDED_DATA = {embedded_data};
const HULL_BASE64 = "{hull_base64}";
const HULL_ROTATION_MATRIX = [
  [0.0, 0.0, -1.0, 0.0],
  [0.0, 1.0, 0.0, 0.0],
  [1.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 1.0],
];

let dataset = null;
let samples = [];
let scene, camera, renderer, controls;
let boomLine, tipSphere, mastSphere;
let currentIndex = 0;
let playing = false;
let lastStep = 0;

const sceneMount = document.getElementById("scene");
const scrub = document.getElementById("scrub");
const plot = document.getElementById("plot");

function buildHullMatrix4() {{
  const m = new THREE.Matrix4();
  m.set(
    HULL_ROTATION_MATRIX[0][0], HULL_ROTATION_MATRIX[0][1], HULL_ROTATION_MATRIX[0][2], HULL_ROTATION_MATRIX[0][3],
    HULL_ROTATION_MATRIX[1][0], HULL_ROTATION_MATRIX[1][1], HULL_ROTATION_MATRIX[1][2], HULL_ROTATION_MATRIX[1][3],
    HULL_ROTATION_MATRIX[2][0], HULL_ROTATION_MATRIX[2][1], HULL_ROTATION_MATRIX[2][2], HULL_ROTATION_MATRIX[2][3],
    HULL_ROTATION_MATRIX[3][0], HULL_ROTATION_MATRIX[3][1], HULL_ROTATION_MATRIX[3][2], HULL_ROTATION_MATRIX[3][3]
  );
  return m;
}}

function base64ToArrayBuffer(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}}

function initScene() {{
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0f14);
  camera = new THREE.PerspectiveCamera(48, 1, 0.01, 1000);
  camera.position.set(4.2, 2.8, 4.2);
  renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  sceneMount.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(-1.4, 0, 0);
  controls.update();

  scene.add(new THREE.AmbientLight(0xffffff, 0.72));
  const dir = new THREE.DirectionalLight(0xffffff, 0.9);
  dir.position.set(5, 8, 5);
  scene.add(dir);
  scene.add(new THREE.GridHelper(10, 10, 0x28485a, 0x172938));
  scene.add(new THREE.AxesHelper(0.9));

  mastSphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.055, 24, 24),
    new THREE.MeshPhongMaterial({{ color: 0x8bc5ff, emissive: 0x13283a }})
  );
  const mount = dataset.metadata.mast_mount_point_m;
  mastSphere.position.set(mount[0], mount[1], mount[2]);
  scene.add(mastSphere);

  tipSphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.06, 24, 24),
    new THREE.MeshPhongMaterial({{ color: 0x65e0c2, emissive: 0x103a31 }})
  );
  scene.add(tipSphere);

  boomLine = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({{ color: 0x65e0c2, linewidth: 4 }})
  );
  scene.add(boomLine);

  try {{
    const geometry = new THREE.STLLoader().parse(base64ToArrayBuffer(HULL_BASE64));
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshPhongMaterial({{
        color: 0x88ccff,
        specular: 0x112233,
        shininess: 120,
        transparent: true,
        opacity: 0.92,
      }})
    );
    mesh.scale.set(0.01, 0.01, 0.01);
    mesh.applyMatrix4(buildHullMatrix4());
    mesh.position.set(-2.974, 0, 0);
    scene.add(mesh);
  }} catch (err) {{
    console.error("Could not load embedded STL", err);
  }}

  const resize = () => {{
    const w = sceneMount.clientWidth;
    const h = sceneMount.clientHeight;
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }};
  resize();
  new ResizeObserver(resize).observe(sceneMount);
}}

function drawPlot() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = plot.getBoundingClientRect();
  plot.width = Math.max(1, Math.floor(rect.width * dpr));
  plot.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = plot.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  if (!samples.length) return;
  const vals = samples.map(s => s.boom_azimuth_deg);
  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const pad = Math.max(5, (maxV - minV) * 0.08);
  const lo = minV - pad;
  const hi = maxV + pad;
  const xFor = i => samples.length === 1 ? w / 2 : (i / (samples.length - 1)) * (w - 24) + 12;
  const yFor = v => h - 12 - ((v - lo) / Math.max(1e-6, hi - lo)) * (h - 24);

  ctx.strokeStyle = "#2b3a47";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(12, h - 12);
  ctx.lineTo(w - 12, h - 12);
  ctx.stroke();

  ctx.strokeStyle = "#65e0c2";
  ctx.lineWidth = 2;
  ctx.beginPath();
  vals.forEach((v, i) => {{
    const x = xFor(i);
    const y = yFor(v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }});
  ctx.stroke();

  const x = xFor(currentIndex);
  ctx.strokeStyle = "#ffc76b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, 8);
  ctx.lineTo(x, h - 8);
  ctx.stroke();

  ctx.fillStyle = "#a9b8c3";
  ctx.font = "12px Segoe UI";
  ctx.fillText(`${{lo.toFixed(1)}} deg`, 14, h - 18);
  ctx.fillText(`${{hi.toFixed(1)}} deg`, 14, 18);
}}

function updateBoom() {{
  if (!samples.length) return;
  const s = samples[currentIndex];
  const mount = dataset.metadata.mast_mount_point_m;
  const tip = [
    mount[0] + s.boom_vector_x_m,
    mount[1] + s.boom_vector_y_m,
    mount[2] + s.boom_vector_z_m,
  ];
  boomLine.geometry.dispose();
  boomLine.geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(mount[0], mount[1], mount[2]),
    new THREE.Vector3(tip[0], tip[1], tip[2]),
  ]);
  tipSphere.position.set(tip[0], tip[1], tip[2]);
  scrub.value = String(currentIndex);
  document.getElementById("timeValue").textContent = `${{s.timestamp_sec.toFixed(2)}} s`;
  document.getElementById("angleValue").textContent = `${{s.boom_azimuth_deg.toFixed(2)}} deg`;
  document.getElementById("frameValue").textContent = String(s.frame_index);
  document.getElementById("modelValue").textContent = dataset.metadata.model.model_label;
  document.getElementById("indexValue").textContent = `${{currentIndex + 1}} / ${{samples.length}}`;
  drawPlot();
}}

function animate(ts) {{
  requestAnimationFrame(animate);
  if (playing && ts - lastStep > 100) {{
    lastStep = ts;
    currentIndex = (currentIndex + 1) % samples.length;
    updateBoom();
  }}
  controls.update();
  renderer.render(scene, camera);
}}

async function main() {{
  dataset = EMBEDDED_DATA;
  samples = dataset.samples || [];
  scrub.max = String(Math.max(0, samples.length - 1));
  initScene();
  updateBoom();
  requestAnimationFrame(animate);
}}

document.getElementById("playBtn").addEventListener("click", () => {{
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "Pause" : "Play";
}});
document.getElementById("prevBtn").addEventListener("click", () => {{
  currentIndex = Math.max(0, currentIndex - 1);
  updateBoom();
}});
document.getElementById("nextBtn").addEventListener("click", () => {{
  currentIndex = Math.min(samples.length - 1, currentIndex + 1);
  updateBoom();
}});
scrub.addEventListener("input", () => {{
  currentIndex = Number(scrub.value) || 0;
  updateBoom();
}});
window.addEventListener("keydown", (ev) => {{
  if (ev.key === " ") {{ playing = !playing; ev.preventDefault(); }}
  if (ev.key === "ArrowLeft") {{ currentIndex = Math.max(0, currentIndex - 1); updateBoom(); }}
  if (ev.key === "ArrowRight") {{ currentIndex = Math.min(samples.length - 1, currentIndex + 1); updateBoom(); }}
}});

main().catch(err => {{
  console.error(err);
  document.body.innerHTML = `<pre style="padding:24px;color:#ffb5b5;">${{err.stack || err}}</pre>`;
}});
</script>
</body>
</html>
"""


def write_html(path: Path, data_obj: dict[str, Any], hull_stl_path: Path) -> None:
    hull_base64 = base64.b64encode(hull_stl_path.read_bytes()).decode("ascii")
    path.write_text(html_template(data_obj, hull_base64), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a GoPro video and render a hull boom-angle timeline")
    parser.add_argument("--video", type=str, required=True, help="GoPro MP4")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Angle model checkpoint")
    parser.add_argument("--model-type", choices=["auto", "pilot", "sail"], default="auto")
    parser.add_argument("--arch", choices=["pilotnet", "boomlite"], default="pilotnet")
    parser.add_argument("--roi-config", type=str, default=str(DEFAULT_ROI_CONFIG))
    parser.add_argument("--roi", type=int, nargs=4, default=None, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Prediction sampling rate")
    parser.add_argument("--every-sec", type=float, default=1.0, help="Used only if --sample-fps <= 0")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="How many sampled frames to preprocess and predict per model call")
    parser.add_argument("--read-mode", choices=["sequential", "seek"], default="sequential",
                        help="sequential is usually much faster for video; seek is useful for sparse manual frames")
    parser.add_argument("--frame-cache", choices=["none", "ram"], default="none",
                        help="Optionally cache sampled raw frames in RAM before inference")
    parser.add_argument("--frame-cache-max-gb", type=float, default=4.0,
                        help="Safety budget for --frame-cache ram")
    parser.add_argument("--boom-length-m", type=float, default=DEFAULT_BOOM_LENGTH_M)
    parser.add_argument("--elevation-deg", type=float, default=0.0)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--invert-angle", action="store_true")
    parser.add_argument("--hull-stl", type=str, default=str(DEFAULT_HULL_STL))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--name", type=str, default="", help="Output name prefix")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    rows, metadata = predict_timeline(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.name.strip() or Path(args.video).stem
    csv_path = out_dir / f"{prefix}_boom_angle_timeline.csv"
    json_path = out_dir / f"{prefix}_boom_angle_timeline.json"
    html_path = out_dir / f"{prefix}_boom_angle_timeline.html"
    hull_out = out_dir / "Hull.stl"

    write_csv(csv_path, rows)
    write_json(json_path, rows, metadata)
    shutil.copyfile(args.hull_stl, hull_out)
    write_html(html_path, {"metadata": metadata, "samples": rows}, Path(args.hull_stl))

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved HTML: {html_path}")
    print("Open the HTML file in a browser to scrub the hull boom timeline.")


if __name__ == "__main__":
    main()
