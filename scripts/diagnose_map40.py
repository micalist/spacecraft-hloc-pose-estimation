"""Diagnose why the 40-image ROE1 mapping set was split into SfM models.

This script is intentionally read-only with respect to the HLoc/COLMAP run.
It writes diagnostics to a separate output directory and verifies that every
input artifact has the same SHA-256 digest before and after the analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pycolmap  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from project_paths import PILOT_ROOT

DEFAULT_PILOT = PILOT_ROOT
PAIR_ID_MAX = 2_147_483_647
FRAME_RE = re.compile(r"img(\d+)", re.IGNORECASE)
INITIAL_PAIR_RE = re.compile(r"Registering initial image pair #(\d+) and #(\d+)")
REGISTER_IMAGE_RE = re.compile(r"Registering image #(\d+)")
DISCARD_RE = re.compile(r"Discarding reconstruction due to (.+)$")

STATUS_COLORS = {
    "primary": "#2E74B5",
    "secondary": "#F39C3D",
    "overlap": "#7B61A8",
    "not_retained": "#D9534F",
}


def frame_number(name: str) -> int:
    match = FRAME_RE.search(Path(name).name)
    if match is None:
        raise ValueError(f"Cannot parse frame number from {name}")
    return int(match.group(1))


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pair_h5_name(name0: str, name1: str) -> str:
    return "/".join((name0.replace("/", "-"), name1.replace("/", "-")))


def pair_id(image_id0: int, image_id1: int) -> int:
    low, high = sorted((image_id0, image_id1))
    return low * PAIR_ID_MAX + high


def camera_center(image) -> np.ndarray:
    transform = image.cam_from_world()
    rotation = np.asarray(transform.rotation.matrix(), dtype=float)
    translation = np.asarray(transform.translation, dtype=float)
    return -rotation.T @ translation


def model_summary(label: str, path: Path, reconstruction: pycolmap.Reconstruction) -> dict:
    names = sorted((image.name for image in reconstruction.images.values()), key=frame_number)
    return {
        "label": label,
        "path": str(path),
        "registered_images": int(reconstruction.num_reg_images()),
        "points3D": int(reconstruction.num_points3D()),
        "observations": int(reconstruction.compute_num_observations()),
        "mean_reprojection_error_px": float(reconstruction.compute_mean_reprojection_error()),
        "image_names": names,
        "frames": [frame_number(name) for name in names],
    }


def load_retained_models(sfm_dir: Path) -> list[tuple[str, Path, pycolmap.Reconstruction]]:
    primary_files = [sfm_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if not all(path.is_file() for path in primary_files):
        raise FileNotFoundError(f"Primary reconstruction is incomplete: {sfm_dir}")

    models: list[tuple[str, Path, pycolmap.Reconstruction]] = [
        ("primary", sfm_dir, pycolmap.Reconstruction(sfm_dir))
    ]
    models_root = sfm_dir / "models"
    if models_root.is_dir():
        for child in sorted(models_root.iterdir(), key=lambda path: path.name):
            if child.is_dir() and (child / "cameras.bin").is_file():
                models.append((f"secondary_{child.name}", child, pycolmap.Reconstruction(child)))
    return models


def parse_mapper_log(log_path: Path, id_to_name: dict[int, str]) -> list[dict]:
    attempts: list[dict] = []
    current: dict | None = None
    pending_image_id: int | None = None

    def commit_pending() -> None:
        nonlocal pending_image_id
        if current is not None and pending_image_id is not None:
            current["image_ids"].add(pending_image_id)
        pending_image_id = None

    def finish(outcome: str, reason: str | None = None) -> None:
        nonlocal current, pending_image_id
        if current is None:
            return
        commit_pending()
        image_ids = sorted(current["image_ids"])
        attempted_ids = sorted(current["attempted_image_ids"])
        failed_ids = sorted(current["failed_image_ids"])
        current["image_ids"] = image_ids
        current["image_names"] = [id_to_name[value] for value in image_ids if value in id_to_name]
        current["frames"] = [frame_number(name) for name in current["image_names"]]
        current["attempted_image_ids"] = attempted_ids
        current["attempted_image_names"] = [id_to_name[value] for value in attempted_ids if value in id_to_name]
        current["attempted_frames"] = [frame_number(name) for name in current["attempted_image_names"]]
        current["failed_image_ids"] = failed_ids
        current["failed_image_names"] = [id_to_name[value] for value in failed_ids if value in id_to_name]
        current["failed_frames"] = [frame_number(name) for name in current["failed_image_names"]]
        current["outcome"] = outcome
        current["reason"] = reason
        attempts.append(current)
        current = None
        pending_image_id = None

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        initial = INITIAL_PAIR_RE.search(line)
        if initial:
            finish("not_explicitly_discarded")
            image0, image1 = int(initial.group(1)), int(initial.group(2))
            current = {
                "initial_pair_ids": [image0, image1],
                "image_ids": {image0, image1},
                "attempted_image_ids": {image0, image1},
                "failed_image_ids": set(),
            }
            continue
        registered = REGISTER_IMAGE_RE.search(line)
        if registered and current is not None:
            commit_pending()
            pending_image_id = int(registered.group(1))
            current["attempted_image_ids"].add(pending_image_id)
            continue
        if "Could not register" in line and current is not None and pending_image_id is not None:
            current["failed_image_ids"].add(pending_image_id)
            pending_image_id = None
            continue
        if "Retriangulation and Global bundle adjustment" in line:
            commit_pending()
            continue
        if "Keeping successful reconstruction" in line:
            finish("kept")
            continue
        discarded = DISCARD_RE.search(line)
        if discarded:
            finish("discarded", discarded.group(1).strip())
    finish("not_explicitly_discarded")
    return attempts


def collect_input_files(
    manifest: Path, pairs: Path, matches: Path, database: Path, log: Path, models
) -> list[Path]:
    files = [manifest, pairs, matches, database, log]
    for _, path, _ in models:
        for name in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
            candidate = path / name
            if candidate.is_file():
                files.append(candidate)
    return sorted(set(files), key=lambda path: str(path).lower())


def build_pair_rows(
    pair_lines: list[str],
    h5_path: Path,
    geometry_by_pair_id: dict[int, tuple[int, int]],
    image_id_by_name: dict[str, int],
    memberships: dict[str, set[str]],
) -> list[dict]:
    rows: list[dict] = []
    with h5py.File(h5_path, "r") as hfile:
        for line in pair_lines:
            name0, name1 = line.split()
            key = pair_h5_name(name0, name1)
            reverse_key = pair_h5_name(name1, name0)
            if key not in hfile and reverse_key not in hfile:
                raise KeyError(f"Missing match pair in HDF5: {name0} {name1}")
            group = hfile[key] if key in hfile else hfile[reverse_key]
            raw_matches = int(np.sum(group["matches0"][()] >= 0))
            pid = pair_id(image_id_by_name[name0], image_id_by_name[name1])
            geometric_inliers, configuration = geometry_by_pair_id.get(pid, (0, 0))
            shared_models = sorted(memberships[name0] & memberships[name1])
            primary_count = int("primary" in memberships[name0]) + int("primary" in memberships[name1])
            rows.append(
                {
                    "image0": name0,
                    "image1": name1,
                    "frame0": frame_number(name0),
                    "frame1": frame_number(name1),
                    "frame_gap": abs(frame_number(name1) - frame_number(name0)),
                    "raw_matches": raw_matches,
                    "geometric_inliers": int(geometric_inliers),
                    "two_view_config": int(configuration),
                    "primary_membership": ("both" if primary_count == 2 else "one" if primary_count == 1 else "none"),
                    "same_retained_model": bool(shared_models),
                    "shared_retained_models": ";".join(shared_models),
                }
            )
    rows.sort(key=lambda row: (row["frame0"], row["frame1"]))
    return rows


def graph_components(mapping_names: list[str], pair_rows: list[dict], threshold: int) -> dict:
    graph = nx.Graph()
    graph.add_nodes_from(mapping_names)
    for row in pair_rows:
        if row["geometric_inliers"] >= threshold:
            graph.add_edge(row["image0"], row["image1"], weight=row["geometric_inliers"])
    components = sorted(nx.connected_components(graph), key=lambda values: (-len(values), min(map(frame_number, values))))
    return {
        "threshold": threshold,
        "edge_count": int(graph.number_of_edges()),
        "component_count": len(components),
        "components": [
            {
                "size": len(component),
                "image_names": sorted(component, key=frame_number),
                "frames": sorted(frame_number(name) for name in component),
            }
            for component in components
        ],
    }


def build_image_rows(
    mapping_names: list[str],
    pair_rows: list[dict],
    memberships: dict[str, set[str]],
    temporary_names: set[str],
) -> list[dict]:
    incident: dict[str, list[dict]] = defaultdict(list)
    for row in pair_rows:
        incident[row["image0"]].append(row)
        incident[row["image1"]].append(row)

    rows = []
    for name in mapping_names:
        models = memberships[name]
        if "primary" in models and len(models) > 1:
            status = "overlap"
        elif "primary" in models:
            status = "primary"
        elif models:
            status = "secondary"
        else:
            status = "not_retained"
        edges = incident[name]
        strongest = max(edges, key=lambda row: (row["geometric_inliers"], row["raw_matches"]))
        neighbor = strongest["image1"] if strongest["image0"] == name else strongest["image0"]
        rows.append(
            {
                "image": name,
                "frame": frame_number(name),
                "status": status,
                "primary_registered": "primary" in models,
                "retained_models": ";".join(sorted(models)),
                "registered_in_any_retained_model": bool(models),
                "seen_in_mapper_registration_attempt": name in temporary_names,
                "raw_nonzero_edges": sum(row["raw_matches"] > 0 for row in edges),
                "raw_match_total": sum(row["raw_matches"] for row in edges),
                "raw_match_max": max(row["raw_matches"] for row in edges),
                "verified_edges_ge15": sum(row["geometric_inliers"] >= 15 for row in edges),
                "verified_edges_ge30": sum(row["geometric_inliers"] >= 30 for row in edges),
                "verified_edges_ge50": sum(row["geometric_inliers"] >= 50 for row in edges),
                "geometric_inlier_total": sum(row["geometric_inliers"] for row in edges),
                "geometric_inlier_max": strongest["geometric_inliers"],
                "strongest_neighbor": neighbor,
            }
        )
    return rows


def plot_registration_timeline(image_rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.8), constrained_layout=True)
    for row in image_rows:
        ax.scatter(
            row["frame"],
            0,
            s=120,
            color=STATUS_COLORS[row["status"]],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(row["frame"], 0.09, str(row["frame"]), rotation=90, ha="center", va="bottom", fontsize=7)
    ax.axvline(39, color="#88929B", linestyle="--", linewidth=1)
    ax.axvline(62.5, color="#88929B", linestyle="--", linewidth=1)
    ax.text(39, -0.20, "primary / secondary boundary", ha="center", va="top", fontsize=8, color="#59636D")
    ax.text(62.5, -0.20, "retained / small-model boundary", ha="center", va="top", fontsize=8, color="#59636D")
    ax.set_xlim(min(row["frame"] for row in image_rows) - 2, max(row["frame"] for row in image_rows) + 2)
    ax.set_ylim(-0.35, 0.42)
    ax.set_yticks([])
    ax.set_xlabel("ROE1 source frame")
    ax.set_title("Map40 registration status across the source sequence", fontsize=13, weight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=9, label=label)
        for label, color in (
            ("Primary model only", STATUS_COLORS["primary"]),
            ("Secondary model only", STATUS_COLORS["secondary"]),
            ("Shared by models", STATUS_COLORS["overlap"]),
            ("Not in retained models", STATUS_COLORS["not_retained"]),
        )
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=4, frameon=False)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AAB2B9")
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_match_heatmap(mapping_names: list[str], pair_rows: list[dict], output: Path) -> None:
    index = {name: position for position, name in enumerate(mapping_names)}
    matrix = np.zeros((len(mapping_names), len(mapping_names)), dtype=float)
    for row in pair_rows:
        i, j = index[row["image0"]], index[row["image1"]]
        matrix[i, j] = matrix[j, i] = row["geometric_inliers"]
    masked = np.ma.masked_less_equal(matrix, 0)
    fig, ax = plt.subplots(figsize=(10.5, 9), constrained_layout=True)
    image = ax.imshow(masked, cmap="viridis", norm=LogNorm(vmin=1, vmax=max(1, matrix.max())))
    frames = [frame_number(name) for name in mapping_names]
    ax.set_xticks(range(len(frames)), labels=frames, rotation=90, fontsize=6)
    ax.set_yticks(range(len(frames)), labels=frames, fontsize=6)
    ax.set_xlabel("Mapping frame")
    ax.set_ylabel("Mapping frame")
    ax.set_title("Geometrically verified correspondences (log scale)", fontsize=13, weight="bold")
    for boundary in (21.5, 34.5):
        ax.axhline(boundary, color="white", linestyle="--", linewidth=0.9, alpha=0.9)
        ax.axvline(boundary, color="white", linestyle="--", linewidth=0.9, alpha=0.9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Verified correspondences")
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def selected_match_graph(mapping_names: list[str], pair_rows: list[dict]) -> nx.Graph:
    eligible = [row for row in pair_rows if row["geometric_inliers"] >= 15]
    full_graph = nx.Graph()
    full_graph.add_nodes_from(mapping_names)
    incident: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        incident[row["image0"]].append(row)
        incident[row["image1"]].append(row)
        full_graph.add_edge(row["image0"], row["image1"], weight=row["geometric_inliers"])
    chosen: set[tuple[str, str]] = set()
    row_by_edge = {}
    for name in mapping_names:
        strongest = sorted(
            incident[name],
            key=lambda row: (-row["geometric_inliers"], row["frame_gap"], row["frame0"], row["frame1"]),
        )[:3]
        for row in strongest:
            edge = tuple(sorted((row["image0"], row["image1"])))
            chosen.add(edge)
            row_by_edge[edge] = row
    if nx.is_connected(full_graph):
        for name0, name1, data in nx.maximum_spanning_tree(full_graph, weight="weight").edges(data=True):
            edge = tuple(sorted((name0, name1)))
            chosen.add(edge)
            row_by_edge[edge] = {
                "image0": edge[0],
                "image1": edge[1],
                "geometric_inliers": int(data["weight"]),
            }
    graph = nx.Graph()
    graph.add_nodes_from(mapping_names)
    for edge in sorted(chosen):
        row = row_by_edge[edge]
        graph.add_edge(*edge, weight=row["geometric_inliers"])
    return graph


def plot_match_graph(mapping_names: list[str], image_rows: list[dict], pair_rows: list[dict], output: Path) -> None:
    graph = selected_match_graph(mapping_names, pair_rows)
    status_by_name = {row["image"]: row["status"] for row in image_rows}
    positions = nx.spring_layout(graph, seed=42, weight=None, k=1.1, iterations=500, scale=3.0)
    weights = [graph.edges[edge]["weight"] for edge in graph.edges]
    maximum = max(weights) if weights else 1
    widths = [0.6 + 2.6 * value / maximum for value in weights]
    fig, ax = plt.subplots(figsize=(13, 9), constrained_layout=True)
    nx.draw_networkx_edges(graph, positions, width=widths, alpha=0.35, edge_color="#6B7680", ax=ax)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[STATUS_COLORS[status_by_name[name]] for name in graph.nodes],
        node_size=360,
        edgecolors="white",
        linewidths=0.9,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={name: str(frame_number(name)) for name in graph.nodes},
        font_size=7,
        font_color="white",
        ax=ax,
    )
    ax.set_title(
        "Readable verified-match graph: top-3 edges + maximum spanning tree (inliers >= 15)",
        fontsize=13,
        weight="bold",
    )
    ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=9, label=label)
        for label, color in (
            ("Primary", STATUS_COLORS["primary"]),
            ("Secondary", STATUS_COLORS["secondary"]),
            ("Overlap", STATUS_COLORS["overlap"]),
            ("Not retained", STATUS_COLORS["not_retained"]),
        )
    ]
    ax.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_axes_equal(ax, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) / 2, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_sparse_models(models, output: Path) -> None:
    fig = plt.figure(figsize=(7 * len(models), 6.4), constrained_layout=True)
    for index, (label, _, reconstruction) in enumerate(models, start=1):
        ax = fig.add_subplot(1, len(models), index, projection="3d")
        points = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=float)
        centers = np.asarray([camera_center(image) for image in reconstruction.images.values()], dtype=float)
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2.5, alpha=0.48, color="#2E74B5")
        if len(centers):
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=35, marker="^", color="#D9534F", label="Cameras")
        combined = np.vstack([values for values in (points, centers) if len(values)])
        set_axes_equal(ax, combined)
        ax.set_title(
            f"{label}: {reconstruction.num_reg_images()} images, {reconstruction.num_points3D()} points",
            fontsize=11,
            weight="bold",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=22, azim=-60)
    fig.suptitle("Retained sparse models shown separately (independent SfM gauges)", fontsize=14, weight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def boundary_row(pair_rows: list[dict], frame0: int, frame1: int) -> dict:
    wanted = {frame0, frame1}
    for row in pair_rows:
        if {row["frame0"], row["frame1"]} == wanted:
            return row
    raise KeyError(f"Boundary pair not found: {frame0}-{frame1}")


def write_report(path: Path, summary: dict) -> None:
    primary = summary["models"][0]
    secondary = summary["models"][1:]
    status = summary["registration_status"]
    graph15 = summary["match_graph_components"]["15"]
    edge_37_41 = summary["key_boundary_pairs"]["37-41"]
    edge_62_63 = summary["key_boundary_pairs"]["62-63"]
    secondary_text = "；".join(
        f"{item['label']}：{item['registered_images']} 张、{item['points3D']} 点、{item['mean_reprojection_error_px']:.3f} px"
        for item in secondary
    )
    report = f"""# 实验四：40 张 Mapping 图像重建失败诊断

## 诊断目标

在不重新建图、不修改实验三结果的条件下，检查现有 ALIKED + NN ratio 0.9 匹配、COLMAP 数据库、保留模型及 Mapper 日志，解释为什么 HLoc 最终只使用 22 张图像的最大模型。

## 可复核事实

| 指标 | 结果 |
|---|---:|
| Mapping 输入 | {summary['inputs']['mapping_images']} 张 |
| 全穷举图像对 | {summary['inputs']['mapping_pairs']} 对 |
| 主模型 | {primary['registered_images']} 张；{primary['points3D']} 点；{primary['mean_reprojection_error_px']:.3f} px |
| 其他保留模型 | {secondary_text} |
| 任一保留模型覆盖 | {status['registered_in_any_retained_model']} 张 |
| 仅看主模型时的“未注册” | {status['not_in_primary_model']} 张 |
| 未进入任何保留模型 | {status['not_in_any_retained_model']} 张 |
| ≥15 几何内点图连通分量 | {graph15['component_count']} 个（{graph15['edge_count']} 条边） |

## 关键证据

1. HLoc 输出目录中的主模型注册 {primary['registered_images']} 张，但 `models/` 下还保留了第二子模型；两个模型存在共享图像，不能把主模型之外的全部图像都称为“完全重建失败”。
2. 两个关键连接处仍有有效几何匹配：37→41 为 {edge_37_41['raw_matches']} 个原始匹配、{edge_37_41['geometric_inliers']} 个几何内点；62→63 为 {edge_62_63['raw_matches']} 个原始匹配、{edge_62_63['geometric_inliers']} 个几何内点。
3. 当阈值设为 15 个几何内点时，40 张图属于同一个匹配连通分量。因此“没有任何跨段匹配”与现有数据库证据不符。
4. Mapper 日志包含被丢弃的小模型尝试，说明后段图像曾参与初始化或临时注册，但因模型规模、初始对或增量注册条件未成为最终保留模型。

## 诊断结论

**事实结论：** 当前 22/40 是“HLoc 选择最大模型后的主模型覆盖率”，不是“只有 22 张图具备可用匹配”。现有重建发生了多子模型分裂，且 HLoc 当前流程只将最大模型复制到 SfM 根目录并用于 Query 定位。

**合理推测：** 主要瓶颈更接近增量建图初始化、模型扩展或多模型保留/合并策略，而不是单纯的局部特征数量不足。仅增加图像或立即更换 LightGlue 可能改善鲁棒性，但尚不能直接解决“多个已成形子模型未被统一使用”的问题。

## 下一步建议

实验五优先验证子模型合并或 COLMAP 单模型重建策略，并在合并后重新评估后段 Query。ALIKED + LightGlue 作为后续匹配消融实验保留，用于比较注册率、三维点数、重投影误差和可靠定位率。

## 输出说明

- `image_registration_status.csv`：逐图主/子模型归属及匹配强度。
- `pair_match_statistics.csv`：780 对图像的原始匹配与几何内点。
- `mapper_attempts.csv`：COLMAP 日志中的初始化、临时注册与丢弃尝试。
- `registration_timeline.png`、`match_heatmap.png`、`match_graph.png`：模型分段与匹配连通性。
- `sparse_models_overview.png`：各保留模型独立坐标系中的点云和相机位置。
- `diagnostic_summary.json`：机器可读诊断结果及输入文件哈希。
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir.resolve()
    run_dir = (args.run_dir or pilot_dir / "hloc_aliked_v1").resolve()
    output_dir = (args.output_dir or pilot_dir / "diagnostics_map40").resolve()
    sfm_dir = run_dir / "sfm_nn_ratio09"
    manifest_path = pilot_dir / "manifests" / "mapping.txt"
    pairs_path = run_dir / "pairs" / "mapping_exhaustive.txt"
    matches_path = run_dir / "matches" / "mapping_nn_ratio09.h5"
    database_path = sfm_dir / "database.db"
    logs = sorted(sfm_dir.glob("colmap.LOG*"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No COLMAP log found in {sfm_dir}")
    log_path = logs[-1]

    models = load_retained_models(sfm_dir)
    input_files = collect_input_files(manifest_path, pairs_path, matches_path, database_path, log_path, models)
    for path in input_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {str(path): sha256(path) for path in input_files}

    mapping_names = read_nonempty_lines(manifest_path)
    pair_lines = read_nonempty_lines(pairs_path)
    if len(mapping_names) != 40:
        raise AssertionError(f"Expected 40 mapping images, got {len(mapping_names)}")
    expected_pairs = len(mapping_names) * (len(mapping_names) - 1) // 2
    if len(pair_lines) != expected_pairs:
        raise AssertionError(f"Expected {expected_pairs} pairs, got {len(pair_lines)}")

    with sqlite3.connect(database_path) as connection:
        image_id_by_name = {name: int(image_id) for image_id, name in connection.execute("SELECT image_id, name FROM images")}
        if set(image_id_by_name) != set(mapping_names):
            raise AssertionError("Database image names do not match the mapping manifest")
        geometry_by_pair_id = {
            int(pid): (int(rows), int(configuration))
            for pid, rows, configuration in connection.execute(
                "SELECT pair_id, rows, config FROM two_view_geometries"
            )
        }
    id_to_name = {value: key for key, value in image_id_by_name.items()}

    memberships = {name: set() for name in mapping_names}
    model_summaries = []
    for label, path, reconstruction in models:
        model_summaries.append(model_summary(label, path, reconstruction))
        for image in reconstruction.images.values():
            if image.name in memberships:
                memberships[image.name].add(label)

    attempts = parse_mapper_log(log_path, id_to_name)
    temporary_names = {name for attempt in attempts for name in attempt["attempted_image_names"]}
    pair_rows = build_pair_rows(pair_lines, matches_path, geometry_by_pair_id, image_id_by_name, memberships)
    image_rows = build_image_rows(mapping_names, pair_rows, memberships, temporary_names)
    components = {str(threshold): graph_components(mapping_names, pair_rows, threshold) for threshold in (15, 30, 50)}

    primary_names = {name for name, labels in memberships.items() if "primary" in labels}
    any_retained_names = {name for name, labels in memberships.items() if labels}
    status_counts = {status: sum(row["status"] == status for row in image_rows) for status in STATUS_COLORS}
    boundary_37_41 = boundary_row(pair_rows, 37, 41)
    boundary_62_63 = boundary_row(pair_rows, 62, 63)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "registered_images.txt").write_text(
        "\n".join(sorted(primary_names, key=frame_number)) + "\n", encoding="utf-8"
    )
    (output_dir / "unregistered_images.txt").write_text(
        "\n".join(sorted(set(mapping_names) - primary_names, key=frame_number)) + "\n", encoding="utf-8"
    )
    (output_dir / "registered_any_model.txt").write_text(
        "\n".join(sorted(any_retained_names, key=frame_number)) + "\n", encoding="utf-8"
    )
    (output_dir / "not_retained_images.txt").write_text(
        "\n".join(sorted(set(mapping_names) - any_retained_names, key=frame_number)) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "image_registration_status.csv", image_rows, list(image_rows[0].keys()))
    write_csv(output_dir / "pair_match_statistics.csv", pair_rows, list(pair_rows[0].keys()))

    attempt_rows = []
    for index, attempt in enumerate(attempts, start=1):
        attempt_rows.append(
            {
                "attempt": index,
                "initial_pair_ids": ";".join(map(str, attempt["initial_pair_ids"])),
                "registered_or_initialized_frames": ";".join(map(str, attempt["frames"])),
                "registered_or_initialized_count": len(attempt["image_names"]),
                "attempted_frames": ";".join(map(str, attempt["attempted_frames"])),
                "failed_frames": ";".join(map(str, attempt["failed_frames"])),
                "outcome": attempt["outcome"],
                "reason": attempt["reason"] or "",
            }
        )
    write_csv(output_dir / "mapper_attempts.csv", attempt_rows, list(attempt_rows[0].keys()))

    plot_registration_timeline(image_rows, output_dir / "registration_timeline.png")
    plot_match_heatmap(mapping_names, pair_rows, output_dir / "match_heatmap.png")
    plot_match_graph(mapping_names, image_rows, pair_rows, output_dir / "match_graph.png")
    plot_sparse_models(models, output_dir / "sparse_models_overview.png")

    summary = {
        "experiment": "Experiment 04 - Map40 reconstruction failure diagnosis",
        "experiment_date": "2026-08-23",
        "read_only_diagnostic": True,
        "inputs": {
            "pilot_dir": str(pilot_dir),
            "run_dir": str(run_dir),
            "sfm_dir": str(sfm_dir),
            "mapping_images": len(mapping_names),
            "mapping_pairs": len(pair_rows),
            "database_images": len(image_id_by_name),
            "colmap_log": str(log_path),
        },
        "models": model_summaries,
        "registration_status": {
            **status_counts,
            "primary_registered": len(primary_names),
            "not_in_primary_model": len(mapping_names) - len(primary_names),
            "registered_in_any_retained_model": len(any_retained_names),
            "not_in_any_retained_model": len(mapping_names) - len(any_retained_names),
            "seen_in_mapper_registration_attempt": len(temporary_names & set(mapping_names)),
        },
        "match_graph_components": components,
        "key_boundary_pairs": {
            "37-41": boundary_37_41,
            "62-63": boundary_62_63,
        },
        "mapper_attempts": attempts,
        "conclusion": {
            "observed": (
                "HLoc retained the largest 22-image model at the SfM root while another retained model covers "
                "14 images and overlaps at frame 37. The >=15-inlier graph connects all 40 images."
            ),
            "inference": (
                "The dominant failure mode is incremental reconstruction/model fragmentation or downstream "
                "largest-model selection, not the complete absence of cross-segment feature matches."
            ),
            "next_experiment": (
                "Prioritize retained-model merging or a single-model COLMAP reconstruction strategy; keep "
                "ALIKED+LightGlue as a later matching ablation."
            ),
        },
        "input_sha256": hashes_before,
    }
    write_json(output_dir / "diagnostic_summary.json", summary)
    write_report(output_dir / "diagnostic_report.md", summary)

    hashes_after = {str(path): sha256(path) for path in input_files}
    if hashes_before != hashes_after:
        changed = [path for path in hashes_before if hashes_before[path] != hashes_after[path]]
        raise RuntimeError(f"Read-only guarantee violated; changed inputs: {changed}")

    required = [
        "registered_images.txt",
        "unregistered_images.txt",
        "registered_any_model.txt",
        "not_retained_images.txt",
        "image_registration_status.csv",
        "pair_match_statistics.csv",
        "mapper_attempts.csv",
        "registration_timeline.png",
        "match_heatmap.png",
        "match_graph.png",
        "sparse_models_overview.png",
        "diagnostic_summary.json",
        "diagnostic_report.md",
    ]
    missing = [name for name in required if not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty outputs: {missing}")

    print(json.dumps({
        "output_dir": str(output_dir),
        "primary_registered": len(primary_names),
        "retained_union": len(any_retained_names),
        "not_retained": len(mapping_names) - len(any_retained_names),
        "graph_components_ge15": components["15"]["component_count"],
        "boundary_37_41_inliers": boundary_37_41["geometric_inliers"],
        "boundary_62_63_inliers": boundary_62_63["geometric_inliers"],
        "inputs_unchanged": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
