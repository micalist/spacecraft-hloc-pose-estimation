"""Run an end-to-end ALIKED HLoc pilot on the ROE1 split."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pycolmap
import torch
from scipy.spatial.transform import Rotation

from project_paths import (
    HLOC_ROOT,
    LIGHTGLUE_ROOT,
    PILOT_ROOT,
    TORCH_CACHE,
    WORKSPACE,
)

PILOT = PILOT_ROOT
IMAGE_DIR = PILOT / "images"
RUN_DIR = PILOT / "hloc_aliked_v1"
MATCH_SUFFIX = "nn_ratio09"
MAPPING_MATCHES = RUN_DIR / "matches" / f"mapping_{MATCH_SUFFIX}.h5"
QUERY_MATCHES = RUN_DIR / "matches" / f"query_to_mapping_{MATCH_SUFFIX}.h5"
SFM_DIR = RUN_DIR / f"sfm_{MATCH_SUFFIX}"
LOCALIZATION_DIR = RUN_DIR / f"localization_{MATCH_SUFFIX}"
EVALUATION_DIR = RUN_DIR / f"evaluation_{MATCH_SUFFIX}"

os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE))
for source_root in (HLOC_ROOT, LIGHTGLUE_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from hloc import extract_features, localize_sfm, match_features, pairs_from_exhaustive  # noqa: E402
from hloc import reconstruction as hloc_reconstruction  # noqa: E402
from hloc.utils.io import list_h5_names, write_poses  # noqa: E402


LOGGER = logging.getLogger("roe1_hloc_pilot")


def configure_logging() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(RUN_DIR / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def read_names(filename: str) -> list[str]:
    path = PILOT / "manifests" / filename
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def camera_config() -> dict:
    return json.loads((PILOT / "config" / "colmap_camera.json").read_text(encoding="utf-8"))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def conservative_roi_mask(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return a conservative rectangular spacecraft ROI from a dark-background frame."""
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu, high = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(high, 8)

    candidates = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        cx, cy = centroids[index]
        near_center = abs(cx - width / 2) < 0.35 * width and abs(cy - height / 2) < 0.40 * height
        if area >= 20 and near_center:
            candidates.append((int(area), int(x), int(y), int(w), int(h)))
    candidates.sort(reverse=True)
    candidates = candidates[:12]

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
    x0 = max(0, x0 - margin_x)
    y0 = max(0, y0 - margin_y)
    x1 = min(width, x1 + margin_x)
    y1 = min(height, y1 + margin_y)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    info = {
        "otsu_threshold": float(otsu),
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "mask_fraction": float(np.mean(mask > 0)),
        "selected_components": len(candidates),
    }
    return mask, info


def make_mask_contact_sheet(names: list[str], mask_root: Path, output: Path) -> None:
    tile_w, tile_h = 300, 188
    columns = 5
    rows = (len(names) + columns - 1) // columns
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    for index, name in enumerate(names):
        image = cv2.imread(str(IMAGE_DIR / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_root / Path(name).with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        outside = mask == 0
        view = image.copy()
        view[outside] = (view[outside] * 0.20).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs):
            cv2.rectangle(view, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 220, 0), 5)
        view = cv2.resize(view, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(view, (0, tile_h - 24), (tile_w, tile_h), (20, 20, 20), -1)
        cv2.putText(view, name, (7, tile_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        row, col = divmod(index, columns)
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = view
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def prepare_inputs() -> dict:
    mapping = read_names("mapping.txt")
    queries = read_names("query.txt")
    all_names = mapping + queries
    mask_root = PILOT / "masks_roi"
    mask_stats = {}
    for name in all_names:
        gray = cv2.imread(str(IMAGE_DIR / name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(IMAGE_DIR / name)
        mask, info = conservative_roi_mask(gray)
        output = mask_root / Path(name).with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), mask):
            raise IOError(f"Could not write {output}")
        mask_stats[name] = info

    lists_dir = RUN_DIR / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    (lists_dir / "mapping.txt").write_text("\n".join(mapping) + "\n", encoding="utf-8")
    (lists_dir / "query.txt").write_text("\n".join(queries) + "\n", encoding="utf-8")
    (lists_dir / "all.txt").write_text("\n".join(all_names) + "\n", encoding="utf-8")

    camera = camera_config()
    camera_line = " ".join(
        [camera["model"], str(camera["width"]), str(camera["height"])]
        + [str(value) for value in camera["params"]]
    )
    query_intrinsics = "\n".join(f"{name} {camera_line}" for name in queries) + "\n"
    (lists_dir / "query_with_intrinsics.txt").write_text(query_intrinsics, encoding="utf-8")

    write_json(RUN_DIR / "mask_stats.json", mask_stats)
    make_mask_contact_sheet(all_names, mask_root, RUN_DIR / "qa" / "mask_contact_sheet.jpg")
    environment = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "opencv": cv2.__version__,
        "pycolmap": pycolmap.__version__,
        "hloc_source": str(HLOC_ROOT),
        "mapping_images": len(mapping),
        "query_images": len(queries),
        "mask_type": "conservative per-frame rectangular spacecraft ROI",
    }
    write_json(RUN_DIR / "environment.json", environment)
    (RUN_DIR / "README.md").write_text(
        "# ROE1 synthetic HLoc/SfM pilot\n\n"
        "This directory contains reproducible ALIKED matching experiments.\n"
        "Masks are conservative per-frame spacecraft ROIs used to reject background keypoints.\n"
        "The mapping set contains 40 images and the query set contains 10 images.\n",
        encoding="utf-8",
    )
    LOGGER.info("Prepared %d mapping and %d query images.", len(mapping), len(queries))
    return environment


def feature_configuration() -> dict:
    conf = copy.deepcopy(extract_features.confs["aliked-n16"])
    conf["output"] = "feats-aliked-n16-n4096-r1600"
    conf["model"]["max_num_keypoints"] = 4096
    conf["model"]["detection_threshold"] = 0.0
    conf["preprocessing"]["resize_max"] = 1600
    return conf


def filtered_feature_path() -> Path:
    return RUN_DIR / "features" / "features-aliked-masked.h5"


def copy_dataset_with_mask(source, target, keep: np.ndarray, n_keypoints: int) -> None:
    data = source[()]
    if source.name.endswith("/descriptors") and data.ndim == 2 and data.shape[-1] == n_keypoints:
        data = data[:, keep]
    elif data.ndim >= 1 and data.shape[0] == n_keypoints and not source.name.endswith("/image_size"):
        data = data[keep]
    created = target.create_dataset(Path(source.name).name, data=data)
    for key, value in source.attrs.items():
        created.attrs[key] = value


def filter_features(raw_path: Path, output_path: Path, names: list[str]) -> dict:
    mask_root = PILOT / "masks_roi"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    stats = {}
    with h5py.File(raw_path, "r") as source, h5py.File(output_path, "w") as target:
        for name in names:
            group = source[name]
            keypoints = group["keypoints"][()]
            mask_path = mask_root / Path(name).with_suffix(".png")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
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
            for _, dataset in group.items():
                copy_dataset_with_mask(dataset, out_group, keep, len(keypoints))
            stats[name] = {
                "before": int(len(keypoints)),
                "after": int(keep.sum()),
                "retained_fraction": float(np.mean(keep)) if len(keep) else 0.0,
                "fallback_central_roi": fallback,
            }
    return stats


def make_feature_contact_sheet(names: list[str], features: Path, output: Path) -> None:
    selected = names[::4]
    tile_w, tile_h = 384, 240
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    with h5py.File(features, "r") as hfile:
        for index, name in enumerate(selected):
            image = cv2.imread(str(IMAGE_DIR / name), cv2.IMREAD_COLOR)
            keypoints = hfile[name]["keypoints"][()]
            sx, sy = tile_w / image.shape[1], tile_h / image.shape[0]
            view = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            for x, y in keypoints:
                cv2.circle(view, (int(round(x * sx)), int(round(y * sy))), 1, (0, 255, 255), -1)
            cv2.rectangle(view, (0, tile_h - 24), (tile_w, tile_h), (20, 20, 20), -1)
            cv2.putText(
                view,
                f"{name} | {len(keypoints)} kp",
                (7, tile_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            row, col = divmod(index, columns)
            sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = view
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def extract_and_filter_features() -> dict:
    names = read_names("mapping.txt") + read_names("query.txt")
    conf = feature_configuration()
    raw = RUN_DIR / "features" / "features-aliked-raw.h5"
    raw.parent.mkdir(parents=True, exist_ok=True)
    extract_features.main(
        conf,
        IMAGE_DIR,
        feature_path=raw,
        image_list=names,
        as_half=False,
        overwrite=False,
    )
    stats = filter_features(raw, filtered_feature_path(), names)
    counts = [item["after"] for item in stats.values()]
    summary = {
        "configuration": conf,
        "per_image": stats,
        "after_min": min(counts),
        "after_median": float(np.median(counts)),
        "after_max": max(counts),
    }
    write_json(RUN_DIR / "features" / "feature_stats.json", summary)
    make_feature_contact_sheet(names, filtered_feature_path(), RUN_DIR / "qa" / "masked_features.jpg")
    LOGGER.info(
        "Masked feature counts: min=%d median=%.1f max=%d",
        summary["after_min"],
        summary["after_median"],
        summary["after_max"],
    )
    return summary


def summarize_matches(path: Path) -> dict:
    counts = []
    with h5py.File(path, "r") as hfile:
        for name in list_h5_names(path):
            matches = hfile[name]["matches0"][()]
            counts.append(int(np.sum(matches >= 0)))
    return {
        "pairs": len(counts),
        "matches_min": min(counts) if counts else 0,
        "matches_median": float(np.median(counts)) if counts else 0.0,
        "matches_max": max(counts) if counts else 0,
        "pairs_with_15_matches": int(sum(value >= 15 for value in counts)),
    }


def match_all_pairs() -> dict:
    lists = RUN_DIR / "lists"
    pairs_dir = RUN_DIR / "pairs"
    matches_dir = RUN_DIR / "matches"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    matches_dir.mkdir(parents=True, exist_ok=True)
    mapping_pairs = pairs_dir / "mapping_exhaustive.txt"
    query_pairs = pairs_dir / "query_to_mapping.txt"
    pairs_from_exhaustive.main(mapping_pairs, image_list=lists / "mapping.txt")
    pairs_from_exhaustive.main(
        query_pairs,
        image_list=lists / "query.txt",
        ref_list=lists / "mapping.txt",
    )

    matcher = copy.deepcopy(match_features.confs["NN-ratio"])
    matcher["output"] = "matches-aliked-NN-mutual-ratio0.9"
    matcher["model"]["ratio_threshold"] = 0.9
    mapping_matches = MAPPING_MATCHES
    query_matches = QUERY_MATCHES
    match_features.main(
        matcher,
        mapping_pairs,
        filtered_feature_path(),
        matches=mapping_matches,
        overwrite=False,
    )
    match_features.main(
        matcher,
        query_pairs,
        filtered_feature_path(),
        matches=query_matches,
        overwrite=False,
    )
    summary = {
        "configuration": matcher,
        "mapping": summarize_matches(mapping_matches),
        "query_to_mapping": summarize_matches(query_matches),
    }
    write_json(matches_dir / f"match_stats_{MATCH_SUFFIX}.json", summary)
    LOGGER.info("Mapping match summary: %s", summary["mapping"])
    LOGGER.info("Query match summary: %s", summary["query_to_mapping"])
    return summary


def reconstruct_mapping() -> pycolmap.Reconstruction:
    sfm_dir = SFM_DIR
    if (sfm_dir / "cameras.bin").exists():
        LOGGER.info("Loading existing reconstruction from %s", sfm_dir)
        return pycolmap.Reconstruction(sfm_dir)

    camera = camera_config()
    image_options = {
        "camera_model": camera["model"],
        "camera_params": ",".join(str(value) for value in camera["params"]),
    }
    mapper_options = {
        "ba_refine_focal_length": False,
        "ba_refine_principal_point": False,
        "ba_refine_extra_params": False,
    }
    model = hloc_reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=IMAGE_DIR,
        pairs=RUN_DIR / "pairs" / "mapping_exhaustive.txt",
        features=filtered_feature_path(),
        matches=MAPPING_MATCHES,
        camera_mode=pycolmap.CameraMode.SINGLE,
        verbose=True,
        image_list=read_names("mapping.txt"),
        image_options=image_options,
        mapper_options=mapper_options,
    )
    if model is None:
        write_json(sfm_dir / "summary.json", {"success": False, "reason": "No model reconstructed"})
        raise RuntimeError("COLMAP did not reconstruct a model")
    summary = {
        "success": True,
        "registered_images": int(model.num_reg_images()),
        "input_images": len(read_names("mapping.txt")),
        "points3D": int(model.num_points3D()),
        "observations": int(model.compute_num_observations()),
        "mean_reprojection_error_px": float(model.compute_mean_reprojection_error()),
        "text_summary": model.summary(),
    }
    write_json(sfm_dir / "summary.json", summary)
    model.export_PLY(sfm_dir / "model.ply")
    LOGGER.info("Reconstruction summary: %s", summary)
    return model


def make_query_camera() -> pycolmap.Camera:
    camera = camera_config()
    return pycolmap.Camera(
        model=camera["model"],
        width=int(camera["width"]),
        height=int(camera["height"]),
        params=np.asarray(camera["params"], dtype=float),
    )


def localize_queries() -> dict:
    sfm = pycolmap.Reconstruction(SFM_DIR)
    db_name_to_id = {image.name: image_id for image_id, image in sfm.images.items()}
    query_names = read_names("query.txt")
    mapping_names = read_names("mapping.txt")
    registered_db_ids = [db_name_to_id[name] for name in mapping_names if name in db_name_to_id]
    camera = make_query_camera()
    localizer = localize_sfm.QueryLocalizer(
        sfm, {"estimation": {"ransac": {"max_error": 4.0}}}
    )
    poses = {}
    stats = {}
    for name in query_names:
        ret, log = localize_sfm.pose_from_cluster(
            localizer,
            name,
            camera,
            registered_db_ids,
            filtered_feature_path(),
            QUERY_MATCHES,
        )
        success = ret is not None
        if success:
            poses[name] = ret["cam_from_world"]
        stats[name] = {
            "success": success,
            "num_raw_2D3D_matches": int(log["num_matches"]),
            "num_unique_2D3D_correspondences": int(len(log["points3D_ids"])),
            "num_inliers": int(ret["num_inliers"]) if success else 0,
        }
        LOGGER.info("Query %s: %s", name, stats[name])

    result_path = LOCALIZATION_DIR / "query_poses.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_poses(poses, result_path, prepend_camera_name=True)
    summary = {
        "localized": len(poses),
        "queries": len(query_names),
        "per_query": stats,
        "ransac_threshold_px": 4.0,
    }
    write_json(LOCALIZATION_DIR / "summary.json", summary)
    return summary


def quaternion_wxyz_to_matrix(values: list[float]) -> np.ndarray:
    w, x, y, z = values
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def ground_truth_pose(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # q_vbs2tango rotates VBS vectors into Tango.  COLMAP uses camera_from_world,
    # hence the required Tango-to-VBS rotation is its transpose.
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
    if not path.exists():
        return poses
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, qw, qx, qy, qz, tx, ty, tz = line.split()
        rotation = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
        translation = np.array([float(tx), float(ty), float(tz)])
        center = -rotation.T @ translation
        poses[name] = (rotation, translation, center)
    return poses


def evaluate_poses() -> dict:
    sfm = pycolmap.Reconstruction(SFM_DIR)
    mapping_labels = json.loads((PILOT / "labels" / "mapping.json").read_text(encoding="utf-8"))
    query_labels = json.loads((PILOT / "labels" / "query.json").read_text(encoding="utf-8"))
    mapping_gt = {item["filename"]: item for item in mapping_labels}
    query_gt = {item["filename"]: item for item in query_labels}

    estimated_centers = []
    true_centers = []
    mapping_rotations = []
    gt_vbs_rotations = []
    used_mapping_names = []
    for _, image in sfm.images.items():
        basename = Path(image.name).name
        if basename not in mapping_gt:
            continue
        rotation, _, center = rigid_components(image.cam_from_world())
        gt_rotation, _, gt_center = ground_truth_pose(mapping_gt[basename])
        estimated_centers.append(center)
        true_centers.append(gt_center)
        mapping_rotations.append(rotation)
        gt_vbs_rotations.append(gt_rotation)
        used_mapping_names.append(image.name)

    estimated_centers_np = np.asarray(estimated_centers)
    true_centers_np = np.asarray(true_centers)
    scale, tango_from_sfm, offset = estimate_similarity(estimated_centers_np, true_centers_np)
    aligned_mapping = (scale * (tango_from_sfm @ estimated_centers_np.T)).T + offset
    mapping_position_residuals = np.linalg.norm(aligned_mapping - true_centers_np, axis=1)

    basis_candidates = []
    for estimated_rotation, gt_vbs_rotation in zip(mapping_rotations, gt_vbs_rotations):
        basis_candidates.append(estimated_rotation @ tango_from_sfm.T @ gt_vbs_rotation.T)
    opencv_from_vbs = project_to_rotation(np.sum(basis_candidates, axis=0))
    mapping_rotation_residuals = [
        rotation_error_degrees(
            estimated_rotation,
            opencv_from_vbs @ gt_vbs_rotation @ tango_from_sfm,
        )
        for estimated_rotation, gt_vbs_rotation in zip(mapping_rotations, gt_vbs_rotations)
    ]

    query_results = read_query_results(LOCALIZATION_DIR / "query_poses.txt")
    localization_summary = json.loads(
        (LOCALIZATION_DIR / "summary.json").read_text(encoding="utf-8")
    )
    rows = []
    for name, (estimated_rotation, _, estimated_center) in query_results.items():
        basename = Path(name).name
        gt_vbs_rotation, gt_target_in_vbs, gt_center = ground_truth_pose(query_gt[basename])
        predicted_center = scale * tango_from_sfm @ estimated_center + offset
        predicted_vbs_rotation = opencv_from_vbs.T @ estimated_rotation @ tango_from_sfm.T
        predicted_target_in_vbs = -predicted_vbs_rotation @ predicted_center
        info = localization_summary["per_query"].get(name, {})
        num_inliers = int(info.get("num_inliers", 0))
        rows.append(
            {
                "name": name,
                "rotation_error_deg": rotation_error_degrees(predicted_vbs_rotation, gt_vbs_rotation),
                "translation_error_m": float(np.linalg.norm(predicted_target_in_vbs - gt_target_in_vbs)),
                "camera_center_error_m": float(np.linalg.norm(predicted_center - gt_center)),
                "translation_relative_error_percent": float(
                    100.0 * np.linalg.norm(predicted_target_in_vbs - gt_target_in_vbs) / np.linalg.norm(gt_target_in_vbs)
                ),
                "num_inliers": num_inliers,
                "reliable_by_15_inliers": num_inliers >= 15,
            }
        )

    evaluation_dir = EVALUATION_DIR
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evaluation_dir / "query_pose_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "name",
            "rotation_error_deg",
            "translation_error_m",
            "camera_center_error_m",
            "translation_relative_error_percent",
            "num_inliers",
            "reliable_by_15_inliers",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rotation_errors = [row["rotation_error_deg"] for row in rows]
    translation_errors = [row["translation_error_m"] for row in rows]
    reliable_rows = [row for row in rows if row["reliable_by_15_inliers"]]
    reliable_rotation_errors = [row["rotation_error_deg"] for row in reliable_rows]
    reliable_translation_errors = [row["translation_error_m"] for row in reliable_rows]
    summary = {
        "alignment": {
            "mapping_images_used": len(used_mapping_names),
            "scale_sfm_to_meters": scale,
            "mapping_center_error_median_m": float(np.median(mapping_position_residuals)),
            "mapping_center_error_max_m": float(np.max(mapping_position_residuals)),
            "mapping_rotation_error_median_deg": float(np.median(mapping_rotation_residuals)),
            "mapping_rotation_error_max_deg": float(np.max(mapping_rotation_residuals)),
            "tango_from_sfm_rotation": tango_from_sfm.tolist(),
            "tango_from_sfm_translation": offset.tolist(),
            "opencv_from_vbs_rotation": opencv_from_vbs.tolist(),
        },
        "query": {
            "localized": len(rows),
            "total": len(query_gt),
            "diagnostic_reliable_inlier_threshold": 15,
            "diagnostic_reliable": len(reliable_rows),
            "diagnostic_reliable_rate": float(len(reliable_rows) / len(query_gt)),
            "rotation_error_median_deg": float(np.median(rotation_errors)) if rows else None,
            "rotation_error_max_deg": float(np.max(rotation_errors)) if rows else None,
            "translation_error_median_m": float(np.median(translation_errors)) if rows else None,
            "translation_error_max_m": float(np.max(translation_errors)) if rows else None,
            "reliable_rotation_error_median_deg": (
                float(np.median(reliable_rotation_errors)) if reliable_rows else None
            ),
            "reliable_rotation_error_max_deg": (
                float(np.max(reliable_rotation_errors)) if reliable_rows else None
            ),
            "reliable_translation_error_median_m": (
                float(np.median(reliable_translation_errors)) if reliable_rows else None
            ),
            "reliable_translation_error_max_m": (
                float(np.max(reliable_translation_errors)) if reliable_rows else None
            ),
            "quality_gate_note": (
                "The 15-inlier gate is a post-run diagnostic threshold, not a preregistered "
                "primary success criterion. Lock it before the formal experiment."
            ),
        },
        "per_query": rows,
        "pose_convention": {
            "q_vbs2tango_true": "scalar-first quaternion rotating VBS vectors into Tango",
            "r_Vo2To_vbs_true": "target-origin vector expressed in VBS",
            "alignment": "Mapping camera centers fit a Sim(3); Mapping rotations estimate a constant VBS-to-OpenCV basis rotation",
        },
    }
    write_json(evaluation_dir / "summary.json", summary)
    LOGGER.info("Evaluation summary: %s", summary["query"])
    return summary


def run_stage(stage: str) -> None:
    start = time.time()
    if stage in ("prepare", "all"):
        prepare_inputs()
    if stage in ("extract", "all"):
        extract_and_filter_features()
    if stage in ("match", "all"):
        match_all_pairs()
    if stage in ("reconstruct", "all"):
        reconstruct_mapping()
    if stage in ("localize", "all"):
        localize_queries()
    if stage in ("evaluate", "all"):
        evaluate_poses()
    LOGGER.info("Stage %s finished in %.1f seconds.", stage, time.time() - start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "extract", "match", "reconstruct", "localize", "evaluate"],
        default="all",
    )
    args = parser.parse_args()
    configure_logging()
    LOGGER.info("Starting stage=%s with Python=%s", args.stage, sys.executable)
    run_stage(args.stage)


if __name__ == "__main__":
    main()
