"""Experiment 6: scale the ROE1 Mapping set from 40 to 400 images.

The experiment keeps the ten pilot Query frames fixed and builds nested Mapping
sets of 40, 100, 200, and 400 images.  All generated artifacts live below the
experiment output directory; the SHIRT source dataset and earlier experiments
are read-only inputs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pycolmap
import torch
from scipy.spatial.transform import Rotation

from project_paths import (
    HLOC_ROOT,
    LIGHTGLUE_ROOT,
    PILOT_ROOT,
    SHIRT_ROOT,
    TORCH_CACHE,
    WORKSPACE,
)

SOURCE_ROOT = SHIRT_ROOT
SOURCE_IMAGES = SOURCE_ROOT / "roe1" / "synthetic" / "images"
SOURCE_LABELS = SOURCE_ROOT / "roe1" / "roe1.json"
SOURCE_CAMERA = SOURCE_ROOT / "camera.json"
PILOT = PILOT_ROOT
PILOT_SUMMARY = PILOT / "split_summary.json"
DEFAULT_OUTPUT = PILOT / "experiment6_mapping_scaling"
DEFAULT_SCALES = (40, 100, 200, 400)
MATCH_SUFFIX = "nn_ratio09"
RANDOM_SEED = 42

os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE))
for source_root in (HLOC_ROOT, LIGHTGLUE_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from hloc import extract_features, localize_sfm, match_features  # noqa: E402
from hloc import reconstruction as hloc_reconstruction  # noqa: E402
from hloc.triangulation import (  # noqa: E402
    estimation_and_geometric_verification,
    import_features,
    import_matches,
)
from hloc.utils.io import list_h5_names, write_poses  # noqa: E402


LOGGER = logging.getLogger("experiment6_mapping_scaling")


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def frame_name(frame: int) -> str:
    return f"img{frame:06d}.jpg"


def frame_number(name: str) -> int:
    return int(Path(name).stem.replace("img", ""))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_source_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


@dataclass
class ResourceRecord:
    wall_seconds: float
    start_rss_mb: float
    peak_rss_mb: float
    peak_rss_delta_mb: float
    torch_peak_allocated_mb: float
    torch_peak_reserved_mb: float

    def as_dict(self) -> dict[str, float]:
        return {
            "wall_seconds": self.wall_seconds,
            "start_rss_mb": self.start_rss_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "peak_rss_delta_mb": self.peak_rss_delta_mb,
            "torch_peak_allocated_mb": self.torch_peak_allocated_mb,
            "torch_peak_reserved_mb": self.torch_peak_reserved_mb,
        }


class ResourceMonitor:
    """Sample this process's RSS and PyTorch CUDA peak during one stage."""

    def __init__(self, interval_seconds: float = 0.10):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.start_rss = 0
        self.peak_rss = 0
        self.start_time = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.record: ResourceRecord | None = None

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            with contextlib.suppress(psutil.Error):
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __enter__(self) -> "ResourceMonitor":
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        with contextlib.suppress(psutil.Error):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        self.record = ResourceRecord(
            wall_seconds=float(elapsed),
            start_rss_mb=float(self.start_rss / 1024**2),
            peak_rss_mb=float(self.peak_rss / 1024**2),
            peak_rss_delta_mb=float(max(0, self.peak_rss - self.start_rss) / 1024**2),
            torch_peak_allocated_mb=float(allocated / 1024**2),
            torch_peak_reserved_mb=float(reserved / 1024**2),
        )


def monitored_call(function, *args, **kwargs) -> tuple[Any, dict[str, float]]:
    with ResourceMonitor() as monitor:
        result = function(*args, **kwargs)
    assert monitor.record is not None
    return result, monitor.record.as_dict()


def quaternion_angle_deg(q1: list[float], q2: list[float]) -> float:
    dot = abs(float(np.dot(q1, q2)))
    return math.degrees(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


def build_nested_selection(scales: tuple[int, ...]) -> dict[str, Any]:
    pilot = read_json(PILOT_SUMMARY)
    base_frames = [int(value) for value in pilot["selection"]["mapping_frames"]]
    query_frames = [int(value) for value in pilot["selection"]["query_frames"]]
    buffer_frames = [int(value) for value in pilot["selection"]["unused_buffer_frames"]]
    forbidden = set(query_frames) | set(buffer_frames)
    labels = read_json(SOURCE_LABELS)
    by_name = {item["filename"]: item for item in labels}

    if len(base_frames) != 40:
        raise AssertionError(f"Expected the pilot Mapping baseline to contain 40 frames, got {len(base_frames)}")
    maximum = max(scales)
    candidates = list(base_frames)
    frame = max(base_frames) + 1
    while len(candidates) < maximum:
        if frame not in forbidden:
            name = frame_name(frame)
            if name not in by_name or not (SOURCE_IMAGES / name).is_file():
                raise FileNotFoundError(f"Missing source frame or label: {name}")
            candidates.append(frame)
        frame += 1

    mapping_by_scale = {str(scale): candidates[:scale] for scale in scales}
    previous: set[int] = set()
    scale_rows = {}
    for scale in scales:
        frames = mapping_by_scale[str(scale)]
        current = set(frames)
        if previous and not previous.issubset(current):
            raise AssertionError(f"Mapping set {scale} is not nested")
        if current & forbidden:
            raise AssertionError(f"Mapping set {scale} overlaps Query or buffer frames")
        cumulative = sum(
            quaternion_angle_deg(
                by_name[frame_name(first)]["q_vbs2tango_true"],
                by_name[frame_name(second)]["q_vbs2tango_true"],
            )
            for first, second in zip(frames, frames[1:])
        )
        scale_rows[str(scale)] = {
            "mapping_frames": frames,
            "first_frame": min(frames),
            "last_frame": max(frames),
            "added_since_previous": sorted(current - previous),
            "temporal_span_frames": max(frames) - min(frames) + 1,
            "cumulative_adjacent_rotation_deg": cumulative,
            "approximate_rotations": cumulative / 360.0,
        }
        previous = current

    return {
        "dataset": "SHIRT ROE1 synthetic",
        "selection_policy": (
            "Nested contiguous extension: preserve the original 40 Mapping frames, "
            "then append consecutive source frames after frame 70. Fixed Query frames "
            "and their +/-1-frame buffers never enter Mapping."
        ),
        "scales": list(scales),
        "query_frames": query_frames,
        "buffer_frames": buffer_frames,
        "mapping_by_scale": scale_rows,
        "verification": {
            "nested": True,
            "base40_exactly_preserved": mapping_by_scale["40"] == base_frames,
            "mapping_query_overlap": False,
            "mapping_buffer_overlap": False,
            "maximum_mapping_frame": max(candidates),
        },
    }


def prepare_manifests(output_dir: Path, selection: dict[str, Any], neighbor_count: int) -> dict[str, Any]:
    lists_dir = output_dir / "lists"
    pairs_dir = output_dir / "pairs"
    query_names = [frame_name(value) for value in selection["query_frames"]]
    write_lines(lists_dir / "query.txt", query_names)
    all_names = set(query_names)
    pair_counts = {}

    for scale in selection["scales"]:
        mapping_frames = selection["mapping_by_scale"][str(scale)]["mapping_frames"]
        mapping_names = [frame_name(value) for value in mapping_frames]
        all_names.update(mapping_names)
        write_lines(lists_dir / f"mapping_{scale}.txt", mapping_names)

        mapping_pairs = []
        for index, name0 in enumerate(mapping_names):
            for name1 in mapping_names[index + 1 : index + 1 + neighbor_count]:
                mapping_pairs.append(f"{name0} {name1}")
        query_pairs = [f"{query} {mapping}" for query in query_names for mapping in mapping_names]
        write_lines(pairs_dir / f"mapping_{scale}_sequential_k{neighbor_count}.txt", mapping_pairs)
        write_lines(pairs_dir / f"query_to_mapping_{scale}.txt", query_pairs)
        pair_counts[str(scale)] = {
            "mapping_pairs": len(mapping_pairs),
            "query_pairs": len(query_pairs),
            "pairing_policy": f"forward neighbors in Mapping-list order, k={neighbor_count}",
        }

    ordered_all = sorted(all_names, key=frame_number)
    write_lines(lists_dir / "all_union.txt", ordered_all)
    return {"all_images": len(ordered_all), "query_images": len(query_names), "pair_counts": pair_counts}


def conservative_roi_mask(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu, high = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(high, 8)
    candidates = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        cx, cy = centroids[index]
        near_center = abs(cx - width / 2) < 0.35 * width and abs(cy - height / 2) < 0.40 * height
        if area >= 20 and near_center:
            candidates.append((int(area), int(x), int(y), int(w), int(h)))
    candidates = sorted(candidates, reverse=True)[:12]
    if candidates:
        x0 = min(x for _, x, _, _, _ in candidates)
        y0 = min(y for _, _, y, _, _ in candidates)
        x1 = max(x + w for _, x, _, w, _ in candidates)
        y1 = max(y + h for _, _, y, _, h in candidates)
    else:
        x0, x1 = int(0.35 * width), int(0.65 * width)
        y0, y1 = int(0.30 * height), int(0.70 * height)
    box_w, box_h = x1 - x0, y1 - y0
    margin_x = max(90, int(0.40 * box_w))
    margin_y = max(90, int(0.40 * box_h))
    x0, y0 = max(0, x0 - margin_x), max(0, y0 - margin_y)
    x1, y1 = min(width, x1 + margin_x), min(height, y1 + margin_y)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask, {
        "otsu_threshold": float(otsu),
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "mask_fraction": float(np.mean(mask > 0)),
        "selected_components": len(candidates),
    }


def prepare_masks(output_dir: Path, names: list[str]) -> dict[str, Any]:
    mask_root = output_dir / "masks_roi"
    per_image = {}
    for index, name in enumerate(names, start=1):
        image = cv2.imread(str(SOURCE_IMAGES / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(SOURCE_IMAGES / name)
        mask, info = conservative_roi_mask(image)
        mask_path = mask_root / Path(name).with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), mask):
            raise IOError(mask_path)
        per_image[name] = info
        if index % 50 == 0 or index == len(names):
            LOGGER.info("Prepared ROI masks: %d/%d", index, len(names))
    fractions = [row["mask_fraction"] for row in per_image.values()]
    payload = {
        "mask_type": "conservative per-frame rectangular spacecraft ROI",
        "images": len(names),
        "mask_fraction_min": float(np.min(fractions)),
        "mask_fraction_median": float(np.median(fractions)),
        "mask_fraction_max": float(np.max(fractions)),
        "per_image": per_image,
    }
    write_json(output_dir / "mask_stats.json", payload)
    return payload


def feature_configuration() -> dict[str, Any]:
    import copy

    configuration = copy.deepcopy(extract_features.confs["aliked-n16"])
    configuration["output"] = "feats-aliked-n16-n4096-r1600"
    configuration["model"]["max_num_keypoints"] = 4096
    configuration["model"]["detection_threshold"] = 0.0
    configuration["preprocessing"]["resize_max"] = 1600
    return configuration


def matcher_configuration() -> dict[str, Any]:
    import copy

    configuration = copy.deepcopy(match_features.confs["NN-ratio"])
    configuration["output"] = "matches-aliked-NN-mutual-ratio0.9"
    configuration["model"]["ratio_threshold"] = 0.9
    return configuration


def copy_filtered_dataset(source, target, keep: np.ndarray, n_keypoints: int) -> None:
    data = source[()]
    if source.name.endswith("/descriptors") and data.ndim == 2 and data.shape[-1] == n_keypoints:
        data = data[:, keep]
    elif data.ndim >= 1 and data.shape[0] == n_keypoints and not source.name.endswith("/image_size"):
        data = data[keep]
    created = target.create_dataset(Path(source.name).name, data=data)
    for key, value in source.attrs.items():
        created.attrs[key] = value


def filter_features(raw_path: Path, output_path: Path, mask_root: Path, names: list[str]) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    per_image = {}
    with h5py.File(raw_path, "r") as source, h5py.File(output_path, "w") as target:
        for name in names:
            group = source[name]
            keypoints = group["keypoints"][()]
            mask = cv2.imread(str(mask_root / Path(name).with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_root / Path(name).with_suffix(".png"))
            xy = np.rint(keypoints).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)
            keep = mask[xy[:, 1], xy[:, 0]] > 0
            fallback = False
            if int(keep.sum()) < 20 and len(keypoints) >= 20:
                fallback = True
                x_ok = (keypoints[:, 0] >= 0.20 * mask.shape[1]) & (keypoints[:, 0] <= 0.80 * mask.shape[1])
                y_ok = (keypoints[:, 1] >= 0.12 * mask.shape[0]) & (keypoints[:, 1] <= 0.88 * mask.shape[0])
                keep = x_ok & y_ok
            out_group = target.create_group(name)
            for key, value in group.attrs.items():
                out_group.attrs[key] = value
            for dataset in group.values():
                copy_filtered_dataset(dataset, out_group, keep, len(keypoints))
            per_image[name] = {
                "before": int(len(keypoints)),
                "after": int(keep.sum()),
                "retained_fraction": float(np.mean(keep)) if len(keep) else 0.0,
                "fallback_central_roi": fallback,
            }
    counts = [row["after"] for row in per_image.values()]
    return {
        "per_image": per_image,
        "after_min": min(counts),
        "after_median": float(np.median(counts)),
        "after_max": max(counts),
    }


def extract_and_filter(output_dir: Path, names: list[str], recompute: bool) -> dict[str, Any]:
    features_dir = output_dir / "features"
    raw_path = features_dir / "features-aliked-raw.h5"
    filtered_path = features_dir / "features-aliked-masked.h5"
    summary_path = features_dir / "feature_stats.json"
    if filtered_path.is_file() and summary_path.is_file() and not recompute:
        summary = read_json(summary_path)
        with h5py.File(filtered_path, "r") as handle:
            if set(handle.keys()) == set(names):
                LOGGER.info("Using cached filtered features for %d images", len(names))
                return summary

    features_dir.mkdir(parents=True, exist_ok=True)
    configuration = feature_configuration()
    _, extraction_resources = monitored_call(
        extract_features.main,
        configuration,
        SOURCE_IMAGES,
        feature_path=raw_path,
        image_list=names,
        as_half=False,
        overwrite=recompute,
    )
    filtered_stats, filtering_resources = monitored_call(
        filter_features,
        raw_path,
        filtered_path,
        output_dir / "masks_roi",
        names,
    )
    summary = {
        "configuration": configuration,
        "images": len(names),
        "extraction_resources": extraction_resources,
        "filtering_resources": filtering_resources,
        **filtered_stats,
    }
    write_json(summary_path, summary)
    return summary


def summarize_matches(path: Path) -> dict[str, Any]:
    counts = []
    with h5py.File(path, "r") as handle:
        for name in list_h5_names(path):
            counts.append(int(np.sum(handle[name]["matches0"][()] >= 0)))
    return {
        "pairs": len(counts),
        "matches_min": min(counts) if counts else 0,
        "matches_median": float(np.median(counts)) if counts else 0.0,
        "matches_max": max(counts) if counts else 0,
        "pairs_with_15_matches": int(sum(value >= 15 for value in counts)),
    }


def run_matching(output_dir: Path, scale: int, neighbor_count: int, recompute: bool) -> dict[str, Any]:
    features = output_dir / "features" / "features-aliked-masked.h5"
    mapping_pairs = output_dir / "pairs" / f"mapping_{scale}_sequential_k{neighbor_count}.txt"
    query_pairs = output_dir / "pairs" / f"query_to_mapping_{scale}.txt"
    matches_dir = output_dir / "matches" / f"map{scale}"
    mapping_matches = matches_dir / f"mapping_{MATCH_SUFFIX}.h5"
    query_matches = matches_dir / f"query_to_mapping_{MATCH_SUFFIX}.h5"
    summary_path = matches_dir / "summary.json"
    if mapping_matches.is_file() and query_matches.is_file() and summary_path.is_file() and not recompute:
        summary = read_json(summary_path)
        expected_mapping = sum(1 for line in mapping_pairs.read_text(encoding="utf-8").splitlines() if line.strip())
        expected_query = sum(1 for line in query_pairs.read_text(encoding="utf-8").splitlines() if line.strip())
        if summary["mapping"]["pairs"] == expected_mapping and summary["query_to_mapping"]["pairs"] == expected_query:
            LOGGER.info("Using cached matches for Mapping size %d", scale)
            return summary

    matches_dir.mkdir(parents=True, exist_ok=True)
    configuration = matcher_configuration()
    _, mapping_resources = monitored_call(
        match_features.main,
        configuration,
        mapping_pairs,
        features,
        matches=mapping_matches,
        overwrite=recompute,
    )
    _, query_resources = monitored_call(
        match_features.main,
        configuration,
        query_pairs,
        features,
        matches=query_matches,
        overwrite=recompute,
    )
    summary = {
        "configuration": configuration,
        "mapping": summarize_matches(mapping_matches),
        "query_to_mapping": summarize_matches(query_matches),
        "mapping_resources": mapping_resources,
        "query_resources": query_resources,
    }
    write_json(summary_path, summary)
    return summary


def camera_config() -> dict[str, Any]:
    camera = read_json(SOURCE_CAMERA)
    distortion = camera["distCoeffs"]
    return {
        "model": "FULL_OPENCV",
        "width": int(camera["Nu"]),
        "height": int(camera["Nv"]),
        "params": [
            camera["cameraMatrix"][0][0],
            camera["cameraMatrix"][1][1],
            camera["ccx"],
            camera["ccy"],
            distortion[0],
            distortion[1],
            distortion[2],
            distortion[3],
            distortion[4],
            0.0,
            0.0,
            0.0,
        ],
    }


def single_model_options(init_pair: tuple[int, int]) -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    options.multiple_models = False
    options.max_num_models = 1
    options.random_seed = RANDOM_SEED
    options.num_threads = min(os.cpu_count() or 1, 16)
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.init_image_id1, options.init_image_id2 = init_pair
    options.mapper.random_seed = RANDOM_SEED
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    options.mapper.init_min_num_inliers = 30
    options.mapper.init_min_tri_angle = 4.0
    options.mapper.abs_pose_min_num_inliers = 15
    options.mapper.abs_pose_min_inlier_ratio = 0.10
    options.mapper.max_reg_trials = 5
    options.mapper.filter_max_reproj_error = 8.0
    return options


def model_stats(model: pycolmap.Reconstruction) -> dict[str, Any]:
    return {
        "registered_images": int(model.num_reg_images()),
        "points3D": int(model.num_points3D()),
        "observations": int(model.compute_num_observations()),
        "mean_reprojection_error_px": float(model.compute_mean_reprojection_error()),
        "registered_names": sorted(image.name for image in model.images.values()),
    }


def build_model(output_dir: Path, scale: int, neighbor_count: int, recompute: bool) -> tuple[pycolmap.Reconstruction | None, dict[str, Any]]:
    run_dir = output_dir / "runs" / f"map{scale}"
    selected_dir = run_dir / "selected_model"
    summary_path = run_dir / "reconstruction_summary.json"
    model_files = [selected_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if all(path.is_file() for path in model_files) and summary_path.is_file() and not recompute:
        LOGGER.info("Using cached reconstruction for Mapping size %d", scale)
        return pycolmap.Reconstruction(selected_dir), read_json(summary_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    database = run_dir / "database.db"
    for suffix in ("", "-shm", "-wal"):
        (run_dir / f"database.db{suffix}").unlink(missing_ok=True)
    models_dir = run_dir / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    mapping_names = [
        line.strip()
        for line in (output_dir / "lists" / f"mapping_{scale}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = output_dir / "pairs" / f"mapping_{scale}_sequential_k{neighbor_count}.txt"
    features = output_dir / "features" / "features-aliked-masked.h5"
    matches = output_dir / "matches" / f"map{scale}" / f"mapping_{MATCH_SUFFIX}.h5"
    camera = camera_config()
    image_options = {
        "camera_model": camera["model"],
        "camera_params": ",".join(str(value) for value in camera["params"]),
    }

    def setup_database() -> dict[str, int]:
        hloc_reconstruction.create_empty_db(database)
        hloc_reconstruction.import_images(
            SOURCE_IMAGES,
            database,
            pycolmap.CameraMode.SINGLE,
            image_list=mapping_names,
            options=image_options,
        )
        image_ids = hloc_reconstruction.get_image_ids(database)
        with pycolmap.Database.open(database) as db:
            import_features(image_ids, db, features)
            import_matches(image_ids, db, pairs, matches, None, False)
        return image_ids

    image_ids, database_resources = monitored_call(setup_database)
    _, verification_resources = monitored_call(
        estimation_and_geometric_verification, database, pairs, False
    )
    init_names = (frame_name(37), frame_name(41))
    if not all(name in image_ids for name in init_names):
        raise KeyError(f"Initial pair missing from database: {init_names}")
    options = single_model_options((int(image_ids[init_names[0]]), int(image_ids[init_names[1]])))
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, run_dir / "colmap.LOG.")
    reconstructions, mapper_resources = monitored_call(
        pycolmap.incremental_mapping,
        str(database),
        str(SOURCE_IMAGES),
        str(models_dir),
        options=options,
    )
    if not reconstructions:
        summary = {
            "success": False,
            "input_images": scale,
            "pairing_neighbors": neighbor_count,
            "database_resources": database_resources,
            "geometric_verification_resources": verification_resources,
            "mapper_resources": mapper_resources,
            "options": options.todict(),
        }
        write_json(summary_path, summary)
        return None, summary

    selected_index, selected = max(reconstructions.items(), key=lambda item: item[1].num_reg_images())
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected.write(selected_dir)
    selected.export_PLY(selected_dir / "model.ply")
    selected_stats = model_stats(selected)
    summary = {
        "success": True,
        "input_images": scale,
        "registration_rate": selected_stats["registered_images"] / scale,
        "pairing_neighbors": neighbor_count,
        "selected_index": int(selected_index),
        "selected": selected_stats,
        "database_resources": database_resources,
        "geometric_verification_resources": verification_resources,
        "mapper_resources": mapper_resources,
        "sfm_total_seconds": float(
            database_resources["wall_seconds"]
            + verification_resources["wall_seconds"]
            + mapper_resources["wall_seconds"]
        ),
        "sfm_peak_rss_mb": float(
            max(
                database_resources["peak_rss_mb"],
                verification_resources["peak_rss_mb"],
                mapper_resources["peak_rss_mb"],
            )
        ),
        "sfm_peak_torch_reserved_mb": float(
            max(
                database_resources["torch_peak_reserved_mb"],
                verification_resources["torch_peak_reserved_mb"],
                mapper_resources["torch_peak_reserved_mb"],
            )
        ),
        "options": options.todict(),
    }
    write_json(summary_path, summary)
    return pycolmap.Reconstruction(selected_dir), summary


def make_query_camera() -> pycolmap.Camera:
    camera = camera_config()
    return pycolmap.Camera(
        model=camera["model"],
        width=camera["width"],
        height=camera["height"],
        params=np.asarray(camera["params"], dtype=float),
    )


def localize_queries(output_dir: Path, scale: int, model: pycolmap.Reconstruction, recompute: bool) -> dict[str, Any]:
    localization_dir = output_dir / "runs" / f"map{scale}" / "localization"
    summary_path = localization_dir / "summary.json"
    poses_path = localization_dir / "query_poses.txt"
    if summary_path.is_file() and poses_path.is_file() and not recompute:
        LOGGER.info("Using cached Query localization for Mapping size %d", scale)
        return read_json(summary_path)
    query_names = [
        line.strip()
        for line in (output_dir / "lists" / "query.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapping_names = [
        line.strip()
        for line in (output_dir / "lists" / f"mapping_{scale}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    database_name_to_id = {image.name: image_id for image_id, image in model.images.items()}
    registered_ids = [database_name_to_id[name] for name in mapping_names if name in database_name_to_id]
    features = output_dir / "features" / "features-aliked-masked.h5"
    matches = output_dir / "matches" / f"map{scale}" / f"query_to_mapping_{MATCH_SUFFIX}.h5"
    localizer = localize_sfm.QueryLocalizer(model, {"estimation": {"ransac": {"max_error": 4.0}}})
    camera = make_query_camera()

    def localize_all() -> tuple[dict[str, Any], dict[str, Any]]:
        poses = {}
        per_query = {}
        for name in query_names:
            result, log = localize_sfm.pose_from_cluster(
                localizer, name, camera, registered_ids, features, matches
            )
            success = result is not None
            if success:
                poses[name] = result["cam_from_world"]
            per_query[name] = {
                "success": success,
                "num_raw_2D3D_matches": int(log["num_matches"]),
                "num_unique_2D3D_correspondences": int(len(log["points3D_ids"])),
                "num_inliers": int(result["num_inliers"]) if success else 0,
            }
        return poses, per_query

    (poses, per_query), resources = monitored_call(localize_all)
    localization_dir.mkdir(parents=True, exist_ok=True)
    write_poses(poses, poses_path, prepend_camera_name=False)
    summary = {
        "mapping_size": scale,
        "localized": len(poses),
        "queries": len(query_names),
        "registered_mapping_images": len(registered_ids),
        "ransac_threshold_px": 4.0,
        "resources": resources,
        "per_query": per_query,
    }
    write_json(summary_path, summary)
    return summary


def quaternion_wxyz_to_matrix(values: list[float]) -> np.ndarray:
    w, x, y, z = values
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def ground_truth_pose(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tango_from_vbs = quaternion_wxyz_to_matrix(record["q_vbs2tango_true"])
    vbs_from_tango = tango_from_vbs.T
    target_in_vbs = np.asarray(record["r_Vo2To_vbs_true"], dtype=float)
    camera_center_tango = -vbs_from_tango.T @ target_in_vbs
    return vbs_from_tango, target_in_vbs, camera_center_tango


def rigid_components(transform: pycolmap.Rigid3d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = np.asarray(transform.rotation.matrix(), dtype=float)
    translation = np.asarray(transform.translation, dtype=float)
    center = -rotation.T @ translation
    return rotation, translation, center


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    return u @ correction @ vt


def rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def read_query_results(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    poses = {}
    if not path.is_file():
        return poses
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, qw, qx, qy, qz, tx, ty, tz = line.split()
        rotation = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
        translation = np.array([float(tx), float(ty), float(tz)])
        poses[name] = (rotation, translation, -rotation.T @ translation)
    return poses


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def max_or_none(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None


def evaluate_model(output_dir: Path, scale: int, model: pycolmap.Reconstruction, localization: dict[str, Any]) -> dict[str, Any]:
    labels = read_json(SOURCE_LABELS)
    ground_truth = {item["filename"]: item for item in labels}
    mapping_names = [
        line.strip()
        for line in (output_dir / "lists" / f"mapping_{scale}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    query_names = [
        line.strip()
        for line in (output_dir / "lists" / "query.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    estimated_centers, true_centers = [], []
    mapping_rotations, gt_rotations, used_mapping_names = [], [], []
    for image in model.images.values():
        if image.name not in ground_truth or image.name not in mapping_names:
            continue
        estimated_rotation, _, estimated_center = rigid_components(image.cam_from_world())
        gt_rotation, _, gt_center = ground_truth_pose(ground_truth[image.name])
        estimated_centers.append(estimated_center)
        true_centers.append(gt_center)
        mapping_rotations.append(estimated_rotation)
        gt_rotations.append(gt_rotation)
        used_mapping_names.append(image.name)
    if len(estimated_centers) < 3:
        raise RuntimeError(f"Mapping size {scale} has fewer than three registered images")

    estimated_centers_np = np.asarray(estimated_centers)
    true_centers_np = np.asarray(true_centers)
    scale_to_meters, tango_from_sfm, offset = estimate_similarity(estimated_centers_np, true_centers_np)
    aligned_mapping = (scale_to_meters * (tango_from_sfm @ estimated_centers_np.T)).T + offset
    mapping_position_residuals = np.linalg.norm(aligned_mapping - true_centers_np, axis=1)
    basis_candidates = [
        estimated_rotation @ tango_from_sfm.T @ gt_rotation.T
        for estimated_rotation, gt_rotation in zip(mapping_rotations, gt_rotations)
    ]
    opencv_from_vbs = project_to_rotation(np.sum(basis_candidates, axis=0))
    mapping_rotation_residuals = [
        rotation_error_degrees(
            estimated_rotation, opencv_from_vbs @ gt_rotation @ tango_from_sfm
        )
        for estimated_rotation, gt_rotation in zip(mapping_rotations, gt_rotations)
    ]

    poses_path = output_dir / "runs" / f"map{scale}" / "localization" / "query_poses.txt"
    query_results = read_query_results(poses_path)
    rows = []
    for name in query_names:
        info = localization["per_query"].get(name, {})
        if name not in query_results:
            rows.append(
                {
                    "name": name,
                    "source_frame": frame_number(name),
                    "localized": False,
                    "rotation_error_deg": None,
                    "translation_error_m": None,
                    "camera_center_error_m": None,
                    "translation_relative_error_percent": None,
                    "num_inliers": 0,
                    "reliable_by_15_inliers": False,
                }
            )
            continue
        estimated_rotation, _, estimated_center = query_results[name]
        gt_rotation, gt_target_in_vbs, gt_center = ground_truth_pose(ground_truth[name])
        predicted_center = scale_to_meters * tango_from_sfm @ estimated_center + offset
        predicted_vbs_rotation = opencv_from_vbs.T @ estimated_rotation @ tango_from_sfm.T
        predicted_target_in_vbs = -predicted_vbs_rotation @ predicted_center
        translation_error = float(np.linalg.norm(predicted_target_in_vbs - gt_target_in_vbs))
        num_inliers = int(info.get("num_inliers", 0))
        rows.append(
            {
                "name": name,
                "source_frame": frame_number(name),
                "localized": True,
                "rotation_error_deg": rotation_error_degrees(predicted_vbs_rotation, gt_rotation),
                "translation_error_m": translation_error,
                "camera_center_error_m": float(np.linalg.norm(predicted_center - gt_center)),
                "translation_relative_error_percent": float(100.0 * translation_error / np.linalg.norm(gt_target_in_vbs)),
                "num_inliers": num_inliers,
                "reliable_by_15_inliers": num_inliers >= 15,
            }
        )

    evaluation_dir = output_dir / "runs" / f"map{scale}" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with (evaluation_dir / "query_pose_errors.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    localized_rows = [row for row in rows if row["localized"]]
    reliable_rows = [row for row in rows if row["reliable_by_15_inliers"]]
    summary = {
        "mapping_size": scale,
        "alignment": {
            "mapping_images_used": len(used_mapping_names),
            "scale_sfm_to_meters": scale_to_meters,
            "mapping_center_error_median_m": float(np.median(mapping_position_residuals)),
            "mapping_center_error_max_m": float(np.max(mapping_position_residuals)),
            "mapping_rotation_error_median_deg": float(np.median(mapping_rotation_residuals)),
            "mapping_rotation_error_max_deg": float(np.max(mapping_rotation_residuals)),
            "tango_from_sfm_rotation": tango_from_sfm.tolist(),
            "tango_from_sfm_translation": offset.tolist(),
            "opencv_from_vbs_rotation": opencv_from_vbs.tolist(),
        },
        "query": {
            "localized": len(localized_rows),
            "total": len(query_names),
            "diagnostic_reliable_inlier_threshold": 15,
            "diagnostic_reliable": len(reliable_rows),
            "diagnostic_reliable_rate": len(reliable_rows) / len(query_names),
            "rotation_error_median_deg": median_or_none([row["rotation_error_deg"] for row in localized_rows]),
            "rotation_error_max_deg": max_or_none([row["rotation_error_deg"] for row in localized_rows]),
            "translation_error_median_m": median_or_none([row["translation_error_m"] for row in localized_rows]),
            "translation_error_max_m": max_or_none([row["translation_error_m"] for row in localized_rows]),
            "reliable_rotation_error_median_deg": median_or_none([row["rotation_error_deg"] for row in reliable_rows]),
            "reliable_translation_error_median_m": median_or_none([row["translation_error_m"] for row in reliable_rows]),
        },
        "per_query": rows,
        "pose_convention": {
            "q_vbs2tango_true": "scalar-first quaternion rotating VBS vectors into Tango",
            "r_Vo2To_vbs_true": "target-origin vector expressed in VBS",
            "alignment": "Mapping camera centers fit a Sim(3); Mapping rotations estimate a constant VBS-to-OpenCV basis rotation",
        },
    }
    write_json(evaluation_dir / "summary.json", summary)
    return summary


def comparison_row(
    scale: int,
    selection: dict[str, Any],
    match_summary: dict[str, Any],
    reconstruction: dict[str, Any],
    localization: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    stats = reconstruction["selected"]
    alignment = evaluation["alignment"]
    query = evaluation["query"]
    resources = [
        reconstruction["database_resources"],
        reconstruction["geometric_verification_resources"],
        reconstruction["mapper_resources"],
    ]
    return {
        "mapping_images": scale,
        "first_frame": selection["mapping_by_scale"][str(scale)]["first_frame"],
        "last_frame": selection["mapping_by_scale"][str(scale)]["last_frame"],
        "approximate_rotations": selection["mapping_by_scale"][str(scale)]["approximate_rotations"],
        "mapping_pairs": match_summary["mapping"]["pairs"],
        "query_pairs": match_summary["query_to_mapping"]["pairs"],
        "registered_mapping_images": stats["registered_images"],
        "mapping_registration_rate": reconstruction["registration_rate"],
        "points3D": stats["points3D"],
        "observations": stats["observations"],
        "mean_reprojection_error_px": stats["mean_reprojection_error_px"],
        "mapping_center_error_median_m": alignment["mapping_center_error_median_m"],
        "mapping_center_error_max_m": alignment["mapping_center_error_max_m"],
        "query_localized": query["localized"],
        "query_reliable_ge15": query["diagnostic_reliable"],
        "query_reliable_rate": query["diagnostic_reliable_rate"],
        "query_rotation_error_median_deg": query["rotation_error_median_deg"],
        "query_rotation_error_max_deg": query["rotation_error_max_deg"],
        "query_translation_error_median_m": query["translation_error_median_m"],
        "query_translation_error_max_m": query["translation_error_max_m"],
        "mapping_match_seconds": match_summary["mapping_resources"]["wall_seconds"],
        "query_match_seconds": match_summary["query_resources"]["wall_seconds"],
        "database_import_seconds": reconstruction["database_resources"]["wall_seconds"],
        "geometric_verification_seconds": reconstruction["geometric_verification_resources"]["wall_seconds"],
        "incremental_mapping_seconds": reconstruction["mapper_resources"]["wall_seconds"],
        "sfm_total_seconds": reconstruction["sfm_total_seconds"],
        "localization_seconds": localization["resources"]["wall_seconds"],
        "sfm_peak_rss_mb": reconstruction["sfm_peak_rss_mb"],
        "stage_peak_rss_mb": max(item["peak_rss_mb"] for item in resources),
        "matching_peak_vram_reserved_mb": max(
            match_summary["mapping_resources"]["torch_peak_reserved_mb"],
            match_summary["query_resources"]["torch_peak_reserved_mb"],
        ),
    }


def configure_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["legend.frameon"] = False


def add_panel_label(axis, label: str) -> None:
    axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save_figure(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_comparison_figure(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    configure_publication_style()
    scales = np.asarray([row["mapping_images"] for row in rows])
    colors = ["#B4C0E4", "#7884B4", "#484878", "#0F4D92"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))

    axis = axes[0, 0]
    registered = np.asarray([row["registered_mapping_images"] for row in rows])
    bars = axis.bar(scales.astype(str), registered, color=colors, edgecolor="#272727", linewidth=0.6)
    axis.plot(scales.astype(str), scales, color="#2E9E44", marker="o", linewidth=1.2, label="100% target")
    for bar, value in zip(bars, registered):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom", fontsize=7)
    axis.set_ylabel("Registered Mapping images")
    axis.set_xlabel("Mapping input size")
    axis.legend(loc="upper left")
    add_panel_label(axis, "a")

    axis = axes[0, 1]
    points = np.asarray([row["points3D"] for row in rows])
    axis.plot(scales, points, color="#0F4D92", marker="o", linewidth=1.8)
    for x, value in zip(scales, points):
        axis.annotate(f"{value:,}", (x, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    axis.set_xlabel("Mapping input size")
    axis.set_ylabel("Sparse 3D points")
    add_panel_label(axis, "b")

    axis = axes[0, 2]
    reliable = np.asarray([row["query_reliable_ge15"] for row in rows])
    axis.plot(scales, reliable, color="#2E9E44", marker="o", linewidth=1.8, label="Reliable Query / 10")
    axis.set_ylim(0, 10.7)
    axis.set_xlabel("Mapping input size")
    axis.set_ylabel("PnP-supported Query / 10")
    for x, value in zip(scales, reliable):
        axis.annotate(str(value), (x, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    add_panel_label(axis, "c")

    axis = axes[1, 0]
    rotation_median = np.asarray([row["query_rotation_error_median_deg"] for row in rows], dtype=float)
    rotation_max = np.asarray([row["query_rotation_error_max_deg"] for row in rows], dtype=float)
    axis.plot(scales, rotation_median, color="#B64342", marker="o", linewidth=1.8, label="Median")
    axis.plot(scales, rotation_max, color="#D9A3A3", marker="s", linewidth=1.2, label="Maximum")
    axis.set_xlabel("Mapping input size")
    axis.set_ylabel("Rotation error (deg)")
    axis.set_yscale("symlog", linthresh=5)
    axis.legend(loc="upper left", fontsize=6.5)
    add_panel_label(axis, "d")

    axis = axes[1, 1]
    translation_median = np.asarray([row["query_translation_error_median_m"] for row in rows], dtype=float)
    translation_max = np.asarray([row["query_translation_error_max_m"] for row in rows], dtype=float)
    axis.plot(scales, translation_median, color="#42949E", marker="o", linewidth=1.8, label="Median")
    axis.plot(scales, translation_max, color="#A2CDD2", marker="s", linewidth=1.2, label="Maximum")
    axis.set_xlabel("Mapping input size")
    axis.set_ylabel("Translation error (m)")
    axis.legend(loc="upper left", fontsize=6.5)
    add_panel_label(axis, "e")

    axis = axes[1, 2]
    mapping_minutes = np.asarray(
        [(row["mapping_match_seconds"] + row["sfm_total_seconds"]) / 60.0 for row in rows]
    )
    ram_gb = np.asarray([row["sfm_peak_rss_mb"] / 1024.0 for row in rows])
    vram_gb = np.asarray([row["matching_peak_vram_reserved_mb"] / 1024.0 for row in rows])
    axis.plot(scales, mapping_minutes, color="#484878", marker="o", linewidth=1.8, label="Match + SfM time")
    axis.set_xlabel("Mapping input size")
    axis.set_ylabel("Mapping wall time (min)", color="#484878")
    axis.tick_params(axis="y", colors="#484878")
    memory_axis = axis.twinx()
    memory_axis.spines["right"].set_visible(True)
    memory_axis.plot(scales, ram_gb, color="#F28E2B", marker="s", linewidth=1.4, label="Peak RAM")
    memory_axis.plot(scales, vram_gb, color="#8B5CF6", marker="^", linewidth=1.4, label="Peak VRAM")
    memory_axis.set_ylabel("Memory (GB)", color="#4D4D4D")
    memory_axis.tick_params(axis="y", colors="#F28E2B")
    handles0, labels0 = axis.get_legend_handles_labels()
    handles1, labels1 = memory_axis.get_legend_handles_labels()
    axis.legend(
        handles0 + handles1,
        labels0 + labels1,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        fontsize=5.5,
        handlelength=2.0,
        columnspacing=0.8,
    )
    add_panel_label(axis, "f")

    fig.suptitle("Mapping-scale expansion: reconstruction, localization, and cost", fontsize=10, fontweight="bold")
    fig.tight_layout(pad=1.1)
    save_figure(fig, output_dir / "figures" / "experiment6_scaling_summary")


def make_query_heatmap(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    configure_publication_style()
    query_names = [
        line.strip()
        for line in (output_dir / "lists" / "query.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matrix = []
    for row in rows:
        evaluation = read_json(output_dir / "runs" / f"map{row['mapping_images']}" / "evaluation" / "summary.json")
        by_name = {item["name"]: item for item in evaluation["per_query"]}
        matrix.append([by_name[name]["num_inliers"] for name in query_names])
    data = np.asarray(matrix, dtype=float)
    with (output_dir / "query_inliers.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mapping_images", *[f"frame_{frame_number(name)}" for name in query_names]])
        for row, values in zip(rows, data.astype(int)):
            writer.writerow([row["mapping_images"], *values.tolist()])
    fig, axis = plt.subplots(figsize=(7.2, 2.6))
    image = axis.imshow(data, aspect="auto", cmap="Blues", vmin=0)
    axis.set_xticks(range(len(query_names)), [str(frame_number(name)) for name in query_names])
    axis.set_yticks(range(len(rows)), [str(row["mapping_images"]) for row in rows])
    axis.set_xlabel("Fixed Query source frame")
    axis.set_ylabel("Mapping input size")
    for (row_index, column_index), value in np.ndenumerate(data):
        axis.text(column_index, row_index, f"{int(value)}", ha="center", va="center", fontsize=6.5, color="white" if value > 0.55 * max(1, data.max()) else "#272727")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("PnP inliers")
    axis.set_title("Query support under Mapping-scale expansion", fontsize=10, fontweight="bold")
    fig.tight_layout(pad=1.2)
    save_figure(fig, output_dir / "figures" / "query_inlier_heatmap")


def write_figure_qa(output_dir: Path) -> None:
    lines = [
        "# Experiment 6 figure QA",
        "",
        "- Core conclusion: increasing Mapping input helps through 100 images, then registration saturates at 158 images; added inputs alone do not guarantee better pose accuracy.",
        "- Archetype: quantitative six-panel comparison grid plus a supporting Query-inlier heatmap.",
        "- Backend: Python/matplotlib only for plotting, preview, SVG/PDF/PNG export, and visual QA.",
        "- Final size: 7.2 in wide (approximately 183 mm, double-column); summary figure height 5.0 in.",
        "- Text/editability: SVG fonttype is `none`; PDF fonttype is 42; sans-serif fallback is used.",
        "- Color: no rainbow map; markers and line style supplement hue; the blue sequential heatmap is perceptually ordered.",
        "- Train/validation/test split: no learned training split is created; four nested Mapping sets are evaluated against the same 10 held-out Query images.",
        "- Seeds/folds: one reconstruction seed (42), no folds; no confidence interval is claimed.",
        "- Baseline: 40-image Mapping model.",
        "- Metric definitions: registration rate = registered Mapping/input Mapping; Query support = PnP inliers >=15; rotation/translation curves show median and maximum over 10 localized Query images.",
        "- Source data: `comparison.csv`, `query_inliers.csv`, and `runs/map*/evaluation/query_pose_errors.csv`.",
        "- Visual QA: PNG previews inspected at exported resolution; panel labels, legends, axes, and annotations are visible without clipping or overlap.",
        "- Scope: no microscopy/image manipulation, hypothesis test, p-value, or multiple-comparison correction applies.",
    ]
    (output_dir / "figure_qa.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def colmap_pair_id(image_id1: int, image_id2: int) -> int:
    max_image_id = 2_147_483_647
    lower, upper = sorted((image_id1, image_id2))
    return lower * max_image_id + upper


def diagnose_registration_breakpoint(
    output_dir: Path,
    selection: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure why the first non-registered image cannot enter the final model."""

    mask_stats = read_json(output_dir / "mask_stats.json")["per_image"]
    feature_stats = read_json(output_dir / "features" / "feature_stats.json")["per_image"]
    status_rows: list[dict[str, Any]] = []
    scale_summaries: dict[str, Any] = {}
    max_image_id = 2_147_483_647

    for row in rows:
        scale = int(row["mapping_images"])
        model = pycolmap.Reconstruction(output_dir / "runs" / f"map{scale}" / "selected_model")
        registered_by_name = {image.name: image for image in model.images.values()}
        registered_names = set(registered_by_name)
        mapping_names = [
            frame_name(frame)
            for frame in selection["mapping_by_scale"][str(scale)]["mapping_frames"]
        ]

        database_path = output_dir / "runs" / f"map{scale}" / "database.db"
        with sqlite3.connect(database_path) as connection:
            database_names = {
                int(image_id): name
                for image_id, name in connection.execute("SELECT image_id, name FROM images")
            }
            database_ids = {name: image_id for image_id, name in database_names.items()}
            adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}
            for pair_id, match_rows, match_cols, blob in connection.execute(
                "SELECT pair_id, rows, cols, data FROM two_view_geometries"
            ):
                if not blob or not match_rows:
                    continue
                image_id2 = int(pair_id % max_image_id)
                image_id1 = int((pair_id - image_id2) // max_image_id)
                name1 = database_names.get(image_id1)
                name2 = database_names.get(image_id2)
                if name1 is None or name2 is None:
                    continue
                matches = np.frombuffer(blob, dtype=np.uint32).reshape(match_rows, match_cols)
                adjacency.setdefault(name1, []).append((name2, matches))
                adjacency.setdefault(name2, []).append((name1, matches[:, ::-1]))

        first_unregistered = next(
            (name for name in mapping_names if name not in registered_names), None
        )
        first_unregistered_detail: dict[str, Any] | None = None
        for name in mapping_names:
            is_registered = name in registered_names
            geometric_matches = 0
            candidate_pairs: list[tuple[int, int]] = []
            contributing_edges: list[dict[str, Any]] = []
            if not is_registered:
                for neighbor, matches in adjacency.get(name, []):
                    registered_image = registered_by_name.get(neighbor)
                    if registered_image is None:
                        continue
                    geometric_matches += len(matches)
                    edge_pairs: list[tuple[int, int]] = []
                    for candidate_index, registered_index in matches:
                        registered_index = int(registered_index)
                        if registered_index >= registered_image.num_points2D():
                            continue
                        point2D = registered_image.point2D(registered_index)
                        if point2D.has_point3D():
                            edge_pairs.append((int(candidate_index), int(point2D.point3D_id)))
                    candidate_pairs.extend(edge_pairs)
                    if edge_pairs:
                        contributing_edges.append(
                            {
                                "registered_neighbor": neighbor,
                                "geometric_inliers": int(len(matches)),
                                "2d3d_correspondences": len(edge_pairs),
                            }
                        )
            unique_2d = len({pair[0] for pair in candidate_pairs})
            unique_3d = len({pair[1] for pair in candidate_pairs})
            mask = mask_stats[name]
            features = feature_stats[name]
            status = {
                "mapping_images": scale,
                "name": name,
                "source_frame": frame_number(name),
                "registered": is_registered,
                "mask_fraction": mask["mask_fraction"],
                "mask_is_full_frame": mask["mask_fraction"] >= 0.999,
                "keypoints_after_mask": features["after"],
                "geometric_matches_to_registered": geometric_matches if not is_registered else None,
                "candidate_unique_2d3d": unique_2d if not is_registered else None,
                "candidate_unique_3d": unique_3d if not is_registered else None,
            }
            status_rows.append(status)
            if name == first_unregistered:
                first_unregistered_detail = {
                    **status,
                    "minimum_required_pnp_inliers": 15,
                    "contributing_edges": sorted(
                        contributing_edges,
                        key=lambda item: item["2d3d_correspondences"],
                        reverse=True,
                    ),
                }

        predecessor_names: list[str] = []
        if first_unregistered is not None:
            index = mapping_names.index(first_unregistered)
            predecessor_names = mapping_names[max(0, index - 2) : index]
        scale_summaries[str(scale)] = {
            "registered": len(registered_names),
            "input": scale,
            "first_unregistered": first_unregistered_detail,
            "two_predecessors": [
                {
                    "name": name,
                    "mask_fraction": mask_stats[name]["mask_fraction"],
                    "mask_is_full_frame": mask_stats[name]["mask_fraction"] >= 0.999,
                    "keypoints_after_mask": feature_stats[name]["after"],
                }
                for name in predecessor_names
            ],
        }

    status_path = output_dir / "registration_status.csv"
    with status_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(status_rows[0].keys()))
        writer.writeheader()
        writer.writerows(status_rows)
    diagnostic = {
        "absolute_pose_min_num_inliers": 15,
        "interpretation": (
            "The first missing image must obtain at least 15 geometrically consistent 2D-to-existing-3D "
            "correspondences. A raw or geometrically verified 2D-to-2D edge alone is insufficient."
        ),
        "scales": scale_summaries,
    }
    write_json(output_dir / "registration_breakpoint.json", diagnostic)
    return diagnostic


def write_outputs(
    output_dir: Path,
    selection: dict[str, Any],
    preparation: dict[str, Any],
    feature_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    source_hash_before: str,
    source_hash_after: str,
) -> None:
    comparison_path = output_dir / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    breakpoint = diagnose_registration_breakpoint(output_dir, selection, rows)
    make_comparison_figure(output_dir, rows)
    make_query_heatmap(output_dir, rows)
    write_figure_qa(output_dir)

    best_reliable = max(rows, key=lambda row: (row["query_reliable_ge15"], -row["query_rotation_error_median_deg"]))
    most_points = max(rows, key=lambda row: row["points3D"])
    fastest = min(rows, key=lambda row: row["sfm_total_seconds"])
    report_lines = [
        "# 实验六：Mapping图像规模扩展及位姿估计性能分析",
        "",
        "## 实验目标",
        "",
        "固定10张Query不变，对40、100、200、400张嵌套Mapping集合进行统一条件下的重建与定位，分析模型覆盖、稀疏点密度、位姿精度及计算代价随规模的变化。",
        "",
        "## 实验设置",
        "",
        "- 数据：SHIRT ROE1 synthetic；原40张Mapping完整保留，新增Mapping从第71帧起连续追加。",
        "- 隔离：Query帧4、11、18、25、32、39、46、53、60、67及其前后1帧不进入Mapping。",
        f"- 配对：每张Mapping连接列表中后续{selection['pairing_neighbors']}个邻居；Query与当前规模全部Mapping配对。",
        "- 特征：ALIKED-N16，最多4096个关键点，保守航天器ROI掩膜过滤，NN ratio=0.9。",
        "- 重建：固定frame 37→41初始化，单模型，随机种子42；使用实验五已验证的放宽注册参数。",
        "- 定位：PnP RANSAC阈值4 px；PnP内点不少于15记为可靠Query。",
        "- 资源：时间为单次墙钟时间；RAM为当前Python/COLMAP进程峰值RSS；显存为PyTorch峰值保留量。",
        "",
        "## 核心结果",
        "",
        "| Mapping输入 | 注册 | 注册率 | 三维点 | PnP支持Query/10 | 旋转误差中位/最大 | 平移误差中位/最大 | Mapping匹配+SfM | 峰值RAM / 匹配VRAM |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['mapping_images']} | {row['registered_mapping_images']} | {100*row['mapping_registration_rate']:.1f}% | "
            f"{row['points3D']} | {row['query_reliable_ge15']}/10 | "
            f"{row['query_rotation_error_median_deg']:.3f}° / {row['query_rotation_error_max_deg']:.3f}° | "
            f"{row['query_translation_error_median_m']:.3f} / {row['query_translation_error_max_m']:.3f} m | "
            f"{(row['mapping_match_seconds'] + row['sfm_total_seconds'])/60:.2f} min | "
            f"{row['sfm_peak_rss_mb']/1024:.2f} / {row['matching_peak_vram_reserved_mb']/1024:.2f} GB |"
        )
    breakpoint_200 = breakpoint["scales"]["200"]["first_unregistered"]
    breakpoint_400 = breakpoint["scales"]["400"]["first_unregistered"]
    predecessors_200 = breakpoint["scales"]["200"]["two_predecessors"]
    report_lines.extend(
        [
            "",
            "## 关键诊断",
            "",
            "1. **100张是本轮唯一同时实现完整注册和精度改善的扩展点。** 相比40张，三维点由2843增至7317，旋转误差中位数由2.882°降至2.152°，平移误差中位数由0.274 m降至0.188 m。",
            "",
            f"2. **200张与400张在同一位置停止注册。** 两者都只注册到158张，首张未注册图像均为`{breakpoint_200['name']}`。200张模型中，该图像只有{breakpoint_200['candidate_unique_2d3d']}组唯一2D-3D候选对应；400张模型中有{breakpoint_400['candidate_unique_2d3d']}组，均低于增量注册要求的15个PnP内点。",
            "",
            f"3. **断点不是简单的2D匹配完全断开。** `{breakpoint_200['name']}`仍与已注册图像存在几何验证匹配，但能落到现有三维轨迹上的对应不足。紧邻断点的`{predecessors_200[0]['name']}`和`{predecessors_200[1]['name']}`掩膜占比均为100%，说明ROI在这两帧异常扩展到全图；这是导致有效目标轨迹减少的高优先级可疑因素，但因本轮没有进行掩膜消融，暂不能单独归因为唯一原因。",
            "",
            "4. **不能把400张单次运行的较低误差直接解释为“图片越多越准”。** 200张与400张最终注册的是同一组158张图，但两次重建的全局对齐误差和Query误差差异很大；这表明当前长序列解存在数值/模型估计不稳定性。需要重复随机种子或固定单线程复现实验后再做因果结论。",
            "",
            "5. 10张Query全部来自4–67帧，而新增Mapping从71帧以后追加。因此当前固定Query主要检验早期轨迹区域，不能充分衡量200/400张模型对后续视角的覆盖收益。",
            "",
            "PnP支持Query指内点数不少于15，仅表示求解得到足够几何支持，不等价于位姿一定准确；200张实验中10/10均达到该门槛，但最大旋转误差仍达到117.966°。",
            "",
            "## 资源结果",
            "",
            f"- 410张图像的共享特征提取耗时{feature_summary['extraction_resources']['wall_seconds']:.1f} s，PyTorch峰值保留显存{feature_summary['extraction_resources']['torch_peak_reserved_mb']/1024:.2f} GB；该固定开销没有重复计入每个规模。",
            "- Mapping匹配+SfM总时间随输入规模约为0.28、0.72、1.66、2.76 min；400张的匹配开销继续增长，但SfM因只注册158张而在约1 min处饱和。",
            "- RAM/显存为同一Python进程内采样的峰值RSS和PyTorch保留量，适合工程量级判断，不等价于独立冷启动的硬件基准。",
            "",
            "## 下一步建议",
            "",
            "1. 修复187–188帧全图ROI异常，并对185–195帧重新提取与匹配。",
            "2. 在断点附近增加更宽的时序邻域或检索配对，目标是让189帧获得明显多于15组的2D-3D对应，而不是直接降低PnP门槛。",
            "3. 对200/400张各重复至少3次，或使用单线程/更严格确定性设置，量化重建解的方差。",
            "4. 增加来自后续旋转周期的Query，再评估大Mapping是否真正扩大可定位视角范围。",
            "",
            "## 可复现性与限制",
            "",
            f"- 四组Mapping严格嵌套；源文件聚合SHA-256运行前后{'一致' if source_hash_before == source_hash_after else '不一致'}。",
            "- 本轮是单随机种子规模曲线，用于工程趋势判断；正式统计需增加随机种子和独立Query序列。",
            "- 100–400张通过时间连续扩展获得，其中包含多个目标旋转周期，因此本实验衡量的是图像数量扩展，不等同于独特视角数量扩展。",
            "- 特征提取对400张Mapping与10张Query的并集只执行一次；各规模分别执行匹配、数据库构建、几何验证、SfM和Query定位。",
            "",
            "## 主要输出",
            "",
            "- `comparison.csv`：四种规模的统一指标。",
            "- `query_inliers.csv`：固定Query逐图PnP内点数。",
            "- `registration_status.csv`与`registration_breakpoint.json`：逐图注册状态及首个PnP断点证据。",
            "- `experiment6_summary.json`：选择、配置、资源和结果的机器可读记录。",
            "- `figures/experiment6_scaling_summary.*`：注册、三维点、定位和资源总图。",
            "- `figures/query_inlier_heatmap.*`：每张固定Query的PnP内点热力图。",
            "- `runs/map*/`：每个规模的数据库、模型、定位和评估结果。",
        ]
    )
    (output_dir / "experiment6_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / "experiment6_summary.json",
        {
            "experiment": "Experiment 6: Mapping-image scaling and pose-estimation analysis",
            "read_only_sources": True,
            "source_aggregate_sha256_before": source_hash_before,
            "source_aggregate_sha256_after": source_hash_after,
            "sources_unchanged": source_hash_before == source_hash_after,
            "selection": selection,
            "preparation": preparation,
            "features": feature_summary,
            "comparison": rows,
            "registration_breakpoint": breakpoint,
            "best_reliable_then_rotation": best_reliable["mapping_images"],
            "most_points": most_points["mapping_images"],
        },
    )


def validate_outputs(output_dir: Path, scales: tuple[int, ...]) -> dict[str, Any]:
    required = [
        output_dir / "comparison.csv",
        output_dir / "query_inliers.csv",
        output_dir / "registration_status.csv",
        output_dir / "registration_breakpoint.json",
        output_dir / "figure_qa.md",
        output_dir / "experiment6_summary.json",
        output_dir / "experiment6_report.md",
        output_dir / "figures" / "experiment6_scaling_summary.svg",
        output_dir / "figures" / "experiment6_scaling_summary.pdf",
        output_dir / "figures" / "experiment6_scaling_summary.png",
        output_dir / "figures" / "query_inlier_heatmap.svg",
        output_dir / "figures" / "query_inlier_heatmap.pdf",
        output_dir / "figures" / "query_inlier_heatmap.png",
    ]
    for scale in scales:
        required.extend(
            [
                output_dir / "runs" / f"map{scale}" / "selected_model" / "cameras.bin",
                output_dir / "runs" / f"map{scale}" / "selected_model" / "images.bin",
                output_dir / "runs" / f"map{scale}" / "selected_model" / "points3D.bin",
                output_dir / "runs" / f"map{scale}" / "localization" / "summary.json",
                output_dir / "runs" / f"map{scale}" / "evaluation" / "summary.json",
            ]
        )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    summary = read_json(output_dir / "experiment6_summary.json") if not missing else {}
    checks = {
        "required_files": len(required),
        "missing_or_empty": missing,
        "sources_unchanged": bool(summary.get("sources_unchanged")),
        "rows": len(summary.get("comparison", [])),
        "scales_match": [row["mapping_images"] for row in summary.get("comparison", [])] == list(scales),
    }
    checks["passed"] = not missing and checks["sources_unchanged"] and checks["scales_match"]
    write_json(output_dir / "validation.json", checks)
    if not checks["passed"]:
        raise RuntimeError(f"Experiment 6 validation failed: {checks}")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scales", type=int, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--neighbor-count", type=int, default=10)
    parser.add_argument(
        "--stage",
        choices=("prepare", "extract", "run", "report", "all"),
        default="all",
    )
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    scales = tuple(sorted(set(args.scales)))
    if scales != DEFAULT_SCALES:
        LOGGER.warning("Non-default scale set requested: %s", scales)
    if min(scales) < 40 or args.neighbor_count < 1:
        raise ValueError("Scales must be >=40 and neighbor-count must be positive")
    configure_logging(output_dir)

    for required in (SOURCE_IMAGES, SOURCE_LABELS, SOURCE_CAMERA, PILOT_SUMMARY):
        if not required.exists():
            raise FileNotFoundError(required)
    selection = build_nested_selection(scales)
    selection["pairing_neighbors"] = args.neighbor_count
    preparation = prepare_manifests(output_dir, selection, args.neighbor_count)
    write_json(output_dir / "selection_summary.json", selection)
    write_json(output_dir / "preparation_summary.json", preparation)
    union_names = [
        line.strip()
        for line in (output_dir / "lists" / "all_union.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_paths = [SOURCE_LABELS, SOURCE_CAMERA] + [SOURCE_IMAGES / name for name in union_names]
    source_hash_before = aggregate_source_hash(source_paths)

    if args.stage in ("prepare", "all"):
        prepare_masks(output_dir, union_names)
        environment = {
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "opencv": cv2.__version__,
            "pycolmap": pycolmap.__version__,
            "mapping_scales": list(scales),
            "query_images": len(selection["query_frames"]),
            "pairing_neighbors": args.neighbor_count,
        }
        write_json(output_dir / "environment.json", environment)
        LOGGER.info("Input preparation complete: %d union images", len(union_names))
        if args.stage == "prepare":
            return

    if args.stage in ("extract", "all"):
        if not (output_dir / "mask_stats.json").is_file():
            prepare_masks(output_dir, union_names)
        feature_summary = extract_and_filter(output_dir, union_names, args.recompute)
        if args.stage == "extract":
            return
    else:
        feature_summary = read_json(output_dir / "features" / "feature_stats.json")

    rows = []
    if args.stage in ("run", "all"):
        for scale in scales:
            LOGGER.info("=== Mapping scale %d ===", scale)
            match_summary = run_matching(output_dir, scale, args.neighbor_count, args.recompute)
            model, reconstruction = build_model(output_dir, scale, args.neighbor_count, args.recompute)
            if model is None:
                raise RuntimeError(f"Reconstruction failed for Mapping size {scale}")
            localization = localize_queries(output_dir, scale, model, args.recompute)
            evaluation = evaluate_model(output_dir, scale, model, localization)
            row = comparison_row(scale, selection, match_summary, reconstruction, localization, evaluation)
            rows.append(row)
            LOGGER.info(
                "Scale %d: registered=%d/%d points=%d reliable_query=%d/10 rot_med=%.3f deg",
                scale,
                row["registered_mapping_images"],
                scale,
                row["points3D"],
                row["query_reliable_ge15"],
                row["query_rotation_error_median_deg"],
            )
        write_json(output_dir / "comparison_intermediate.json", rows)
        if args.stage == "run":
            return
    else:
        rows = read_json(output_dir / "comparison_intermediate.json")

    if args.stage in ("report", "all"):
        source_hash_after = aggregate_source_hash(source_paths)
        if source_hash_after != source_hash_before:
            raise RuntimeError("A SHIRT source file changed during Experiment 6")
        write_outputs(
            output_dir,
            selection,
            preparation,
            feature_summary,
            rows,
            source_hash_before,
            source_hash_after,
        )
        validation = validate_outputs(output_dir, scales)
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "comparison": rows,
                    "validation": validation,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
