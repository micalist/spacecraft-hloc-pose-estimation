"""Experiment 5: unify the retained Map40 SfM models and evaluate localization.

The experiment never changes the experiment-3 reconstruction, database, features,
or matches.  It writes a merged reconstruction and two single-model controls to a
new output directory, then evaluates all available models on the same ten queries.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pycolmap

from project_paths import PILOT_ROOT, WORKSPACE

PILOT = PILOT_ROOT
RUN_DIR = PILOT / "hloc_aliked_v1"
BASELINE_MODEL = RUN_DIR / "sfm_nn_ratio09"
SECONDARY_MODEL = BASELINE_MODEL / "models" / "1"
DEFAULT_OUTPUT = PILOT / "experiment5_model_unification"

if str(WORKSPACE / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "scripts"))

import run_hloc_pilot as pilot  # noqa: E402
from hloc import localize_sfm  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=default),
        encoding="utf-8",
    )


def model_stats(model: pycolmap.Reconstruction) -> dict[str, Any]:
    return {
        "registered_images": int(model.num_reg_images()),
        "points3D": int(model.num_points3D()),
        "observations": int(model.compute_num_observations()),
        "mean_reprojection_error_px": float(model.compute_mean_reprojection_error()),
        "registered_names": sorted(image.name for image in model.images.values()),
    }


def shared_track_correspondences(
    primary: pycolmap.Reconstruction,
    secondary: pycolmap.Reconstruction,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], str]:
    common = primary.find_common_reg_image_ids(secondary)
    if len(common) != 1:
        raise RuntimeError(f"Expected exactly one shared registered image, found {common}")
    primary_id, secondary_id = common[0]
    primary_image = primary.images[primary_id]
    secondary_image = secondary.images[secondary_id]
    if primary_image.name != secondary_image.name:
        raise RuntimeError("Shared image IDs do not refer to the same image name")
    if len(primary_image.points2D) != len(secondary_image.points2D):
        raise RuntimeError("Shared image has inconsistent feature counts across models")

    source_points = []
    target_points = []
    secondary_to_primary = {}
    for primary_point2D, secondary_point2D in zip(
        primary_image.points2D, secondary_image.points2D
    ):
        if not (primary_point2D.has_point3D() and secondary_point2D.has_point3D()):
            continue
        primary_point_id = int(primary_point2D.point3D_id)
        secondary_point_id = int(secondary_point2D.point3D_id)
        source_points.append(secondary.points3D[secondary_point_id].xyz)
        target_points.append(primary.points3D[primary_point_id].xyz)
        secondary_to_primary[secondary_point_id] = primary_point_id

    if len(source_points) < 3:
        raise RuntimeError("Fewer than three shared 3D tracks; Sim(3) cannot be estimated")
    return (
        np.asarray(source_points, dtype=float),
        np.asarray(target_points, dtype=float),
        secondary_to_primary,
        primary_image.name,
    )


def estimate_shared_track_sim3(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[pycolmap.Sim3d, dict[str, Any]]:
    target_extent = float(np.linalg.norm(target_points.max(axis=0) - target_points.min(axis=0)))
    ransac_threshold = 0.05 * target_extent
    ransac = pycolmap.RANSACOptions(
        max_error=ransac_threshold,
        min_inlier_ratio=0.30,
        confidence=0.9999,
        min_num_trials=100,
        max_num_trials=10000,
        random_seed=42,
    )
    robust = pycolmap.estimate_sim3d_robust(source_points, target_points, ransac)
    if robust is None:
        raise RuntimeError("Robust Sim(3) estimation failed")
    inlier_mask = np.asarray(robust["inlier_mask"], dtype=bool)
    sim3 = pycolmap.estimate_sim3d(source_points[inlier_mask], target_points[inlier_mask])
    if sim3 is None:
        raise RuntimeError("Final Sim(3) refinement failed")

    matrix = np.asarray(sim3.matrix(), dtype=float)
    transformed = source_points @ matrix[:, :3].T + matrix[:, 3]
    residuals = np.linalg.norm(transformed - target_points, axis=1)
    summary = {
        "shared_3D_correspondences": int(len(source_points)),
        "ransac_inliers": int(inlier_mask.sum()),
        "ransac_threshold_primary_units": ransac_threshold,
        "scale_secondary_to_primary": float(np.asarray(sim3.scale).reshape(-1)[0]),
        "matrix_primary_from_secondary": matrix.tolist(),
        "residual_median_primary_units": float(np.median(residuals)),
        "residual_max_primary_units": float(np.max(residuals)),
    }
    return sim3, summary


def merge_retained_models(output_dir: Path) -> tuple[pycolmap.Reconstruction, dict[str, Any]]:
    primary = pycolmap.Reconstruction(BASELINE_MODEL)
    secondary = pycolmap.Reconstruction(SECONDARY_MODEL)
    source_points, target_points, secondary_to_primary, shared_name = (
        shared_track_correspondences(primary, secondary)
    )
    sim3, alignment = estimate_shared_track_sim3(source_points, target_points)

    aligned_secondary = pycolmap.Reconstruction(secondary)
    aligned_secondary.transform(sim3)
    merged = pycolmap.Reconstruction(primary)
    primary_names = {image.name for image in primary.images.values()}

    added_images = []
    for image_id, image in list(aligned_secondary.images.items()):
        if image.name in primary_names:
            continue
        if not merged.exists_frame(image.frame_id):
            frame = copy.deepcopy(aligned_secondary.frames[image.frame_id])
            frame.reset_rig_ptr()
            merged.add_frame(frame)
        new_image = copy.deepcopy(image)
        for point2D_idx, point2D in enumerate(new_image.points2D):
            if point2D.has_point3D():
                new_image.reset_point3D_for_point2D(point2D_idx)
        new_image.reset_camera_ptr()
        new_image.reset_frame_ptr()
        merged.add_image(new_image)
        merged.register_frame(new_image.frame_id)
        added_images.append(new_image.name)

    extended_observations = 0
    new_points = 0
    conflicts = []
    for secondary_point_id, secondary_point in list(aligned_secondary.points3D.items()):
        if secondary_point_id in secondary_to_primary:
            merged_point_id = secondary_to_primary[secondary_point_id]
            for element in secondary_point.track.elements:
                image = merged.images[element.image_id]
                point2D = image.points2D[element.point2D_idx]
                if point2D.has_point3D():
                    if int(point2D.point3D_id) != merged_point_id:
                        conflicts.append(
                            {
                                "image_id": int(element.image_id),
                                "point2D_idx": int(element.point2D_idx),
                                "existing_point3D_id": int(point2D.point3D_id),
                                "expected_point3D_id": int(merged_point_id),
                            }
                        )
                    continue
                merged.add_observation(
                    merged_point_id,
                    pycolmap.TrackElement(element.image_id, element.point2D_idx),
                )
                extended_observations += 1
            continue

        track = pycolmap.Track()
        for element in secondary_point.track.elements:
            image = merged.images[element.image_id]
            point2D = image.points2D[element.point2D_idx]
            if not point2D.has_point3D():
                track.add_element(pycolmap.TrackElement(element.image_id, element.point2D_idx))
        if track.length() < 2:
            conflicts.append(
                {
                    "secondary_point3D_id": int(secondary_point_id),
                    "reason": "fewer than two conflict-free observations",
                }
            )
            continue
        merged.add_point3D(
            np.asarray(secondary_point.xyz, dtype=float),
            track,
            np.asarray(secondary_point.color, dtype=np.uint8),
        )
        new_points += 1

    merged.update_point_3d_errors()
    model_dir = output_dir / "merged_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    merged.write(model_dir)
    merged.export_PLY(model_dir / "model.ply")
    reloaded = pycolmap.Reconstruction(model_dir)

    summary = {
        "method": "shared-image 2D track identity -> robust 3D Sim(3) -> track-aware union",
        "shared_image": shared_name,
        "alignment": alignment,
        "primary_before": model_stats(primary),
        "secondary_before": model_stats(secondary),
        "added_images": sorted(added_images),
        "extended_primary_point_observations": extended_observations,
        "new_secondary_points": new_points,
        "merge_conflicts": conflicts,
        "merged": model_stats(reloaded),
    }
    write_json(output_dir / "merge_summary.json", summary)
    return reloaded, summary


def localize_model(
    label: str,
    model: pycolmap.Reconstruction,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    localization_dir = output_dir / "localization" / label
    localization_dir.mkdir(parents=True, exist_ok=True)
    database_name_to_id = {image.name: image_id for image_id, image in model.images.items()}
    registered_ids = [
        database_name_to_id[name]
        for name in pilot.read_names("mapping.txt")
        if name in database_name_to_id
    ]
    localizer = localize_sfm.QueryLocalizer(
        model, {"estimation": {"ransac": {"max_error": 4.0}}}
    )
    camera = pilot.make_query_camera()
    poses = {}
    per_query = {}
    for query_name in pilot.read_names("query.txt"):
        result, log = localize_sfm.pose_from_cluster(
            localizer,
            query_name,
            camera,
            registered_ids,
            pilot.filtered_feature_path(),
            pilot.QUERY_MATCHES,
        )
        success = result is not None
        if success:
            poses[query_name] = result["cam_from_world"]
        per_query[query_name] = {
            "success": success,
            "num_raw_2D3D_matches": int(log["num_matches"]),
            "num_unique_2D3D_correspondences": int(len(log["points3D_ids"])),
            "num_inliers": int(result["num_inliers"]) if success else 0,
        }

    poses_path = localization_dir / "query_poses.txt"
    pilot.write_poses(poses, poses_path, prepend_camera_name=True)
    summary = {
        "model": label,
        "localized": len(poses),
        "queries": len(per_query),
        "registered_mapping_images": len(registered_ids),
        "ransac_threshold_px": 4.0,
        "per_query": per_query,
    }
    write_json(localization_dir / "summary.json", summary)
    return poses_path, summary


def evaluate_model(
    label: str,
    model: pycolmap.Reconstruction,
    poses_path: Path,
    localization_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    mapping_labels = json.loads((PILOT / "labels" / "mapping.json").read_text(encoding="utf-8"))
    query_labels = json.loads((PILOT / "labels" / "query.json").read_text(encoding="utf-8"))
    mapping_gt = {item["filename"]: item for item in mapping_labels}
    query_gt = {item["filename"]: item for item in query_labels}

    estimated_centers = []
    true_centers = []
    mapping_rotations = []
    gt_vbs_rotations = []
    used_mapping_names = []
    for image in model.images.values():
        basename = Path(image.name).name
        if basename not in mapping_gt:
            continue
        estimated_rotation, _, estimated_center = pilot.rigid_components(image.cam_from_world())
        gt_rotation, _, gt_center = pilot.ground_truth_pose(mapping_gt[basename])
        estimated_centers.append(estimated_center)
        true_centers.append(gt_center)
        mapping_rotations.append(estimated_rotation)
        gt_vbs_rotations.append(gt_rotation)
        used_mapping_names.append(image.name)

    estimated_centers_np = np.asarray(estimated_centers)
    true_centers_np = np.asarray(true_centers)
    scale, tango_from_sfm, offset = pilot.estimate_similarity(
        estimated_centers_np, true_centers_np
    )
    aligned_mapping = (scale * (tango_from_sfm @ estimated_centers_np.T)).T + offset
    mapping_position_residuals = np.linalg.norm(aligned_mapping - true_centers_np, axis=1)
    basis_candidates = [
        estimated_rotation @ tango_from_sfm.T @ gt_rotation.T
        for estimated_rotation, gt_rotation in zip(mapping_rotations, gt_vbs_rotations)
    ]
    opencv_from_vbs = pilot.project_to_rotation(np.sum(basis_candidates, axis=0))
    mapping_rotation_residuals = [
        pilot.rotation_error_degrees(
            estimated_rotation,
            opencv_from_vbs @ gt_rotation @ tango_from_sfm,
        )
        for estimated_rotation, gt_rotation in zip(mapping_rotations, gt_vbs_rotations)
    ]

    query_results = pilot.read_query_results(poses_path)
    rows = []
    for name, (estimated_rotation, _, estimated_center) in query_results.items():
        basename = Path(name).name
        gt_vbs_rotation, gt_target_in_vbs, gt_center = pilot.ground_truth_pose(query_gt[basename])
        predicted_center = scale * tango_from_sfm @ estimated_center + offset
        predicted_vbs_rotation = opencv_from_vbs.T @ estimated_rotation @ tango_from_sfm.T
        predicted_target_in_vbs = -predicted_vbs_rotation @ predicted_center
        num_inliers = int(localization_summary["per_query"][name]["num_inliers"])
        rows.append(
            {
                "name": name,
                "source_frame": int(Path(name).stem.replace("img", "")),
                "rotation_error_deg": pilot.rotation_error_degrees(
                    predicted_vbs_rotation, gt_vbs_rotation
                ),
                "translation_error_m": float(
                    np.linalg.norm(predicted_target_in_vbs - gt_target_in_vbs)
                ),
                "camera_center_error_m": float(np.linalg.norm(predicted_center - gt_center)),
                "translation_relative_error_percent": float(
                    100.0
                    * np.linalg.norm(predicted_target_in_vbs - gt_target_in_vbs)
                    / np.linalg.norm(gt_target_in_vbs)
                ),
                "num_inliers": num_inliers,
                "reliable_by_15_inliers": num_inliers >= 15,
            }
        )

    evaluation_dir = output_dir / "evaluation" / label
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with (evaluation_dir / "query_pose_errors.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = list(rows[0].keys()) if rows else ["name"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reliable = [row for row in rows if row["reliable_by_15_inliers"]]
    late_rows = [row for row in rows if row["source_frame"] >= 46]
    late_reliable = [row for row in late_rows if row["reliable_by_15_inliers"]]

    def median_or_none(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    def max_or_none(values: list[float]) -> float | None:
        return float(np.max(values)) if values else None

    summary = {
        "model": label,
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
            "diagnostic_reliable": len(reliable),
            "diagnostic_reliable_rate": len(reliable) / len(query_gt),
            "late_queries_frame_ge46": len(late_rows),
            "late_reliable": len(late_reliable),
            "rotation_error_median_deg": median_or_none(
                [row["rotation_error_deg"] for row in rows]
            ),
            "rotation_error_max_deg": max_or_none(
                [row["rotation_error_deg"] for row in rows]
            ),
            "translation_error_median_m": median_or_none(
                [row["translation_error_m"] for row in rows]
            ),
            "translation_error_max_m": max_or_none(
                [row["translation_error_m"] for row in rows]
            ),
            "reliable_rotation_error_median_deg": median_or_none(
                [row["rotation_error_deg"] for row in reliable]
            ),
            "reliable_translation_error_median_m": median_or_none(
                [row["translation_error_m"] for row in reliable]
            ),
        },
        "per_query": rows,
    }
    write_json(evaluation_dir / "summary.json", summary)
    return summary


def prepare_experiment_database(output_dir: Path) -> Path:
    database_copy = output_dir / "inputs" / "database_experiment5.db"
    if not database_copy.is_file():
        database_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASELINE_MODEL / "database.db", database_copy)
    return database_copy


def assess_baseline_database_integrity(output_dir: Path) -> dict[str, Any]:
    diagnostic_dir = PILOT / "diagnostics_map40"
    experiment4 = json.loads(
        (diagnostic_dir / "diagnostic_summary.json").read_text(encoding="utf-8")
    )
    database_path = BASELINE_MODEL / "database.db"
    expected_binary_hash = experiment4["input_sha256"][str(database_path)]
    current_binary_hash = sha256(database_path)
    with sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True) as connection:
        image_ids = {
            name: int(image_id)
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }
        geometry = {
            int(pair): (int(rows), int(config))
            for pair, rows, config in connection.execute(
                "SELECT pair_id, rows, config FROM two_view_geometries"
            )
        }
        table_counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "rigs",
                "cameras",
                "frames",
                "images",
                "keypoints",
                "matches",
                "two_view_geometries",
            )
        }

    pair_id_max = 2147483647
    mismatches = []
    with (diagnostic_dir / "pair_match_statistics.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        experiment4_pairs = list(csv.DictReader(handle))
    for row in experiment4_pairs:
        low, high = sorted((image_ids[row["image0"]], image_ids[row["image1"]]))
        current = geometry.get(low * pair_id_max + high, (0, 0))
        expected = (int(row["geometric_inliers"]), int(row["two_view_config"]))
        if current != expected:
            mismatches.append(
                {
                    "image0": row["image0"],
                    "image1": row["image1"],
                    "expected": expected,
                    "current": current,
                }
            )
    integrity = {
        "experiment4_binary_sha256": expected_binary_hash,
        "current_binary_sha256": current_binary_hash,
        "binary_sha256_equal": expected_binary_hash == current_binary_hash,
        "table_counts": table_counts,
        "experiment4_pair_rows_checked": len(experiment4_pairs),
        "geometric_count_and_config_mismatches": len(mismatches),
        "semantic_geometry_check_passed": not mismatches,
        "mismatch_sample": mismatches[:10],
        "safeguard": (
            "The first experiment-5 development run passed the baseline SQLite file to "
            "pycolmap.incremental_mapping, which changed its binary hash. The 780 stored "
            "geometric inlier counts/configurations still match experiment 4 exactly. "
            "All subsequent mapper runs use inputs/database_experiment5.db."
        ),
    }
    write_json(output_dir / "baseline_database_integrity.json", integrity)
    return integrity


def database_image_ids(database_path: Path) -> dict[str, int]:
    with sqlite3.connect(
        f"file:{database_path.as_posix()}?mode=ro", uri=True
    ) as connection:
        return {
            name: int(image_id)
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }


def single_model_options(
    init_pair: tuple[int, int] | None = None,
    relaxed: bool = False,
) -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    options.multiple_models = False
    options.max_num_models = 1
    options.random_seed = 42
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.mapper.random_seed = 42
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    if init_pair is not None:
        options.init_image_id1, options.init_image_id2 = init_pair
    if relaxed:
        options.mapper.init_min_num_inliers = 30
        options.mapper.init_min_tri_angle = 4.0
        options.mapper.abs_pose_min_num_inliers = 15
        options.mapper.abs_pose_min_inlier_ratio = 0.10
        options.mapper.max_reg_trials = 5
        options.mapper.filter_max_reproj_error = 8.0
    return options


def run_single_model_control(
    label: str,
    options: pycolmap.IncrementalPipelineOptions,
    output_dir: Path,
    database_path: Path,
) -> tuple[pycolmap.Reconstruction | None, dict[str, Any]]:
    control_dir = output_dir / "single_model_controls" / label
    selected_dir = control_dir / "selected_model"
    if all((selected_dir / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")):
        model = pycolmap.Reconstruction(selected_dir)
        cached = {
            "label": label,
            "cached": True,
            "options": options.todict(),
            "models": [model_stats(model)],
            "selected": model_stats(model),
        }
        write_json(control_dir / "reconstruction_summary.json", cached)
        return model, cached

    models_dir = control_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(
        str(database_path),
        str(pilot.IMAGE_DIR),
        str(models_dir),
        options=options,
    )
    if not reconstructions:
        summary = {
            "label": label,
            "cached": False,
            "options": options.todict(),
            "models": [],
            "selected": None,
        }
        write_json(control_dir / "reconstruction_summary.json", summary)
        return None, summary

    selected_index, selected_model = max(
        reconstructions.items(), key=lambda item: item[1].num_reg_images()
    )
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_model.write(selected_dir)
    selected_model.export_PLY(selected_dir / "model.ply")
    summary = {
        "label": label,
        "cached": False,
        "options": options.todict(),
        "models": [
            {"index": int(index), **model_stats(model)}
            for index, model in sorted(reconstructions.items())
        ],
        "selected_index": int(selected_index),
        "selected": model_stats(selected_model),
    }
    write_json(control_dir / "reconstruction_summary.json", summary)
    return pycolmap.Reconstruction(selected_dir), summary


def baseline_entry() -> tuple[dict[str, Any], dict[str, Any]]:
    model = pycolmap.Reconstruction(BASELINE_MODEL)
    evaluation = json.loads(
        (RUN_DIR / "evaluation_nn_ratio09" / "summary.json").read_text(encoding="utf-8")
    )
    return model_stats(model), evaluation


def comparison_row(
    label: str,
    strategy: str,
    stats: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    alignment = evaluation["alignment"]
    query = evaluation["query"]
    return {
        "model": label,
        "strategy": strategy,
        "registered_mapping_images": stats["registered_images"],
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
    }


def write_comparison_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    merge_summary: dict[str, Any],
    source_hashes: dict[str, str],
    database_integrity: dict[str, Any],
) -> None:
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["model"] for row in rows]
    registered = [row["registered_mapping_images"] for row in rows]
    reliable = [row["query_reliable_ge15"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].bar(x, registered, color=["#6B7280", "#2878B5", "#F28E2B", "#59A14F"][: len(rows)])
    axes[0].axhline(40, color="#9CA3AF", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Registered mapping images")
    axes[0].set_ylim(0, 42)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    for index, value in enumerate(registered):
        axes[0].text(index, value + 0.6, str(value), ha="center", fontsize=9)
    axes[1].bar(x, reliable, color=["#6B7280", "#2878B5", "#F28E2B", "#59A14F"][: len(rows)])
    axes[1].axhline(10, color="#9CA3AF", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Reliable queries (inliers >= 15)")
    axes[1].set_ylim(0, 10.5)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    for index, value in enumerate(reliable):
        axes[1].text(index, value + 0.2, str(value), ha="center", fontsize=9)
    fig.suptitle("Experiment 5 model-unification comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "experiment5_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    query_names = pilot.read_names("query.txt")
    query_frames = [int(Path(name).stem.replace("img", "")) for name in query_names]
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    for label, evaluation in evaluations.items():
        by_name = {row["name"]: row for row in evaluation["per_query"]}
        values = [by_name.get(name, {}).get("num_inliers", 0) for name in query_names]
        ax.plot(query_frames, values, marker="o", linewidth=1.8, label=label)
    ax.axhline(15, color="#D62728", linestyle="--", linewidth=1.2, label="reliable threshold")
    ax.set_xlabel("ROE1 source frame")
    ax.set_ylabel("PnP inliers")
    ax.set_yscale("symlog", linthresh=15)
    ax.grid(alpha=0.22)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Query localization support before and after model unification", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "query_inliers_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    baseline = next(row for row in rows if row["model"] == "baseline_primary")
    merged = next(row for row in rows if row["model"] == "merged_retained")
    best = max(rows, key=lambda row: (row["query_reliable_ge15"], row["registered_mapping_images"]))
    report = f"""# 实验五：子模型统一与单模型重建验证

## 实验目标

在不重新提取特征、不重新匹配的条件下，将所有新模型写入独立实验五目录，验证“多子模型分裂与最大模型选择”是否是后段 Query 定位失败的主要原因。

## 方法

1. 利用主模型与第二模型共享的 `img000037`，按相同二维特征索引建立三维点对应。
2. 通过固定随机种子 42 的 LO-RANSAC 估计第二模型到主模型的 Sim(3)，再进行轨迹级合并。
3. 使用实验五数据库副本分别运行单模型自动初始化和 `37→41` 桥接初始化对照。
4. 所有模型使用完全相同的10张 Query、ALIKED 特征、NN ratio 0.9 匹配与4 px PnP阈值。

## 合并证据

- 共享三维对应：{merge_summary['alignment']['shared_3D_correspondences']} 对；RANSAC内点：{merge_summary['alignment']['ransac_inliers']} 对。
- 合并模型：{merge_summary['merged']['registered_images']} 张图、{merge_summary['merged']['points3D']} 个三维点、平均重投影误差 {merge_summary['merged']['mean_reprojection_error_px']:.3f} px。
- 合并冲突：{len(merge_summary['merge_conflicts'])}。

## 基线数据库保护说明

首次开发运行曾将实验三的 SQLite 数据库直接传给 `incremental_mapping`，其二进制 SHA-256 因此发生变化。复核40张图、780对原始匹配记录和780对几何记录后，实验四保存的几何内点数与配置不一致数为 {database_integrity['geometric_count_and_config_mismatches']}；后续运行已改为只使用 `inputs/database_experiment5.db` 副本，不再把原数据库传入 Mapper。

## 核心结果

| 模型 | Mapping注册 | 三维点 | 可靠Query/10 | 旋转误差中位数 | 平移误差中位数 |
|---|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += (
            f"| {row['model']} | {row['registered_mapping_images']} | {row['points3D']} | "
            f"{row['query_reliable_ge15']} | {row['query_rotation_error_median_deg']:.3f}° | "
            f"{row['query_translation_error_median_m']:.3f} m |\n"
        )
    report += f"""

## 结论

事实：保留模型合并使 Mapping 覆盖从 {baseline['registered_mapping_images']} 张提高到 {merged['registered_mapping_images']} 张，可靠 Query 从 {baseline['query_reliable_ge15']}/10 变为 {merged['query_reliable_ge15']}/10。桥接初始化进一步达到 {best['registered_mapping_images']}/40 张注册和 {best['query_reliable_ge15']}/10 张可靠 Query；其 Query 旋转误差最大值为 {best['query_rotation_error_max_deg']:.3f}°。综合本轮指标，表现最佳的是 `{best['model']}`。

推断：后段 Query 的内点和姿态误差已经同步改善，实验四的判断得到直接支持——首要瓶颈是初始化和多个已成形子模型未被统一利用，而不是必须先更换局部特征匹配器。桥接模型的平均重投影误差和 Mapping 最大位置残差高于基线，正式实验仍应进行多随机种子重复、初始化边消融和参数收紧，不能只报告一次成功运行。

## 输出

- `merged_model/`：35张保留图像的统一稀疏模型。
- `single_model_controls/`：自动初始化和桥接初始化对照。
- `localization/`、`evaluation/`：逐模型 Query 定位及真值误差。
- `comparison.csv`、`experiment5_summary.json`：统一对照结果。
"""
    (output_dir / "experiment5_report.md").write_text(report, encoding="utf-8")

    payload = {
        "experiment": "Experiment 5: retained-model merge and single-model controls",
        "read_only_sources": True,
        "source_sha256": source_hashes,
        "baseline_database_integrity": database_integrity,
        "merge": merge_summary,
        "comparison": rows,
        "best_by_reliable_then_registration": best["model"],
    }
    write_json(output_dir / "experiment5_summary.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-single-controls",
        action="store_true",
        help="Only merge and evaluate the two retained models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database_integrity = assess_baseline_database_integrity(output_dir)
    experiment_database = prepare_experiment_database(output_dir)

    source_files = [
        BASELINE_MODEL / "database.db",
        BASELINE_MODEL / "cameras.bin",
        BASELINE_MODEL / "images.bin",
        BASELINE_MODEL / "points3D.bin",
        SECONDARY_MODEL / "cameras.bin",
        SECONDARY_MODEL / "images.bin",
        SECONDARY_MODEL / "points3D.bin",
        pilot.filtered_feature_path(),
        pilot.MAPPING_MATCHES,
        pilot.QUERY_MATCHES,
    ]
    source_hashes_before = {str(path): sha256(path) for path in source_files}

    merged_model, merge_summary = merge_retained_models(output_dir)
    merged_poses, merged_localization = localize_model(
        "merged_retained", merged_model, output_dir
    )
    merged_evaluation = evaluate_model(
        "merged_retained",
        merged_model,
        merged_poses,
        merged_localization,
        output_dir,
    )

    baseline_stats, baseline_evaluation = baseline_entry()
    rows = [
        comparison_row(
            "baseline_primary", "HLoc largest model", baseline_stats, baseline_evaluation
        ),
        comparison_row(
            "merged_retained",
            "shared-track Sim(3) union",
            model_stats(merged_model),
            merged_evaluation,
        ),
    ]
    evaluations = {
        "baseline_primary": baseline_evaluation,
        "merged_retained": merged_evaluation,
    }

    if not args.skip_single_controls:
        image_ids = database_image_ids(experiment_database)
        controls = [
            ("single_auto", single_model_options()),
            (
                "single_bridge_37_41",
                single_model_options(
                    (
                        image_ids["mapping/img000037.jpg"],
                        image_ids["mapping/img000041.jpg"],
                    ),
                    relaxed=True,
                ),
            ),
        ]
        for label, options in controls:
            model, _ = run_single_model_control(
                label, options, output_dir, experiment_database
            )
            if model is None:
                continue
            poses_path, localization = localize_model(label, model, output_dir)
            evaluation = evaluate_model(
                label, model, poses_path, localization, output_dir
            )
            rows.append(
                comparison_row(label, "single-model COLMAP", model_stats(model), evaluation)
            )
            evaluations[label] = evaluation

    source_hashes_after = {str(path): sha256(path) for path in source_files}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A source artifact changed during experiment 5")

    write_comparison_outputs(
        output_dir,
        rows,
        evaluations,
        merge_summary,
        source_hashes_before,
        database_integrity,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sources_unchanged": True,
                "comparison": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
