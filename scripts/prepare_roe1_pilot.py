"""Create a deterministic ROE1/synthetic pilot split for HLoc/SfM.

The source SHIRT dataset is never modified.  The default split uses a
70-frame contiguous window.  Ten query frames are placed every seven frames;
each query and its immediate temporal neighbours are removed from the mapping
set, leaving exactly 40 mapping frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from project_paths import PILOT_ROOT, SHIRT_ROOT


def quaternion_angle_deg(q1: list[float], q2: list[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_contact_sheet(image_paths: list[Path], output: Path, columns: int) -> None:
    thumb_w, thumb_h, label_h = 240, 150, 22
    rows = math.ceil(len(image_paths) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(image_paths):
        with Image.open(path) as image:
            tile = image.convert("RGB")
            tile.thumbnail((thumb_w, thumb_h))
        col, row = index % columns, index // columns
        x = col * thumb_w + (thumb_w - tile.width) // 2
        y = row * (thumb_h + label_h)
        sheet.paste(tile, (x, y))
        draw.text((col * thumb_w + 4, y + thumb_h + 3), path.stem, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=SHIRT_ROOT,
        help="SHIRT dataset root (default: SHIRT_ROOT or <repo>/shirtv1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PILOT_ROOT,
        help="Pilot output root (default: POSE_PILOT_ROOT or <repo>/pilot_roe1_synthetic_v1)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    source_images = source / "roe1" / "synthetic" / "images"
    labels_path = source / "roe1" / "roe1.json"
    camera_path = source / "camera.json"

    for required in (source_images, labels_path, camera_path):
        if not required.exists():
            raise FileNotFoundError(required)

    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}. Use --force to replace it.")
        if output == source or source in output.parents:
            raise RuntimeError("Refusing to remove the source dataset or a directory inside it.")
        shutil.rmtree(output)

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    by_name = {item["filename"]: item for item in labels}
    if len(by_name) != len(labels):
        raise RuntimeError("Duplicate filenames in roe1.json")

    # Frames 1..70 span about 324 degrees.  This is wide enough to test
    # reconstruction while avoiding the near-duplicate views that appear after
    # one complete target rotation (about 77 frames).
    window_frames = list(range(1, 71))
    query_frames = [4 + 7 * index for index in range(10)]
    excluded_from_mapping = {
        frame + offset for frame in query_frames for offset in (-1, 0, 1)
    }
    mapping_frames = [frame for frame in window_frames if frame not in excluded_from_mapping]
    unused_buffer_frames = sorted(excluded_from_mapping - set(query_frames))

    if len(mapping_frames) != 40 or len(query_frames) != 10:
        raise RuntimeError(
            f"Unexpected split size: mapping={len(mapping_frames)}, query={len(query_frames)}"
        )
    if set(mapping_frames) & set(query_frames):
        raise RuntimeError("Mapping/query overlap detected")

    def filename(frame: int) -> str:
        return f"img{frame:06d}.jpg"

    mapping_names = [filename(frame) for frame in mapping_frames]
    query_names = [filename(frame) for frame in query_frames]
    buffer_names = [filename(frame) for frame in unused_buffer_frames]
    for name in mapping_names + query_names + buffer_names:
        if name not in by_name:
            raise KeyError(f"Missing label: {name}")
        if not (source_images / name).is_file():
            raise FileNotFoundError(source_images / name)

    mapping_dir = output / "images" / "mapping"
    query_dir = output / "images" / "query"
    labels_dir = output / "labels"
    manifests_dir = output / "manifests"
    config_dir = output / "config"
    qa_dir = output / "qa"
    for directory in (mapping_dir, query_dir, labels_dir, manifests_dir, config_dir, qa_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copied: list[tuple[Path, Path]] = []
    for split, names, target_dir in (
        ("mapping", mapping_names, mapping_dir),
        ("query", query_names, query_dir),
    ):
        for name in names:
            src = source_images / name
            dst = target_dir / name
            shutil.copy2(src, dst)
            copied.append((src, dst))

    mapping_labels = [dict(by_name[name], source_frame=int(name[3:9]), split="mapping") for name in mapping_names]
    query_labels = [dict(by_name[name], source_frame=int(name[3:9]), split="query") for name in query_names]
    (labels_dir / "mapping.json").write_text(
        json.dumps(mapping_labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (labels_dir / "query.json").write_text(
        json.dumps(query_labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (manifests_dir / "mapping.txt").write_text(
        "".join(f"mapping/{name}\n" for name in mapping_names), encoding="utf-8"
    )
    (manifests_dir / "query.txt").write_text(
        "".join(f"query/{name}\n" for name in query_names), encoding="utf-8"
    )
    (manifests_dir / "buffer_unused.txt").write_text(
        "".join(f"{name}\n" for name in buffer_names), encoding="utf-8"
    )
    shutil.copy2(camera_path, config_dir / "camera_shirt.json")

    dist = camera["distCoeffs"]
    full_opencv = {
        "model": "FULL_OPENCV",
        "width": camera["Nu"],
        "height": camera["Nv"],
        "params": [
            camera["cameraMatrix"][0][0],
            camera["cameraMatrix"][1][1],
            camera["ccx"],
            camera["ccy"],
            dist[0],
            dist[1],
            dist[2],
            dist[3],
            dist[4],
            0.0,
            0.0,
            0.0,
        ],
        "param_order": [
            "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"
        ],
        "source": "SHIRT camera.json; OpenCV five-coefficient distortion embedded in COLMAP FULL_OPENCV",
    }
    (config_dir / "colmap_camera.json").write_text(
        json.dumps(full_opencv, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_q = {name: by_name[name]["q_vbs2tango_true"] for name in by_name}
    mapping_temporal_gaps = [b - a for a, b in zip(mapping_frames, mapping_frames[1:])]
    mapping_angle_gaps = [
        quaternion_angle_deg(all_q[filename(a)], all_q[filename(b)])
        for a, b in zip(mapping_frames, mapping_frames[1:])
    ]
    query_nearest = []
    for frame in query_frames:
        qname = filename(frame)
        candidates = [
            (
                quaternion_angle_deg(all_q[qname], all_q[mname]),
                abs(frame - mframe),
                mframe,
            )
            for mframe, mname in zip(mapping_frames, mapping_names)
        ]
        angle_deg, frame_gap, nearest_frame = min(candidates)
        query_nearest.append(
            {
                "query_frame": frame,
                "nearest_mapping_frame": nearest_frame,
                "temporal_gap_frames": frame_gap,
                "attitude_gap_deg": angle_deg,
            }
        )

    source_copy_hash_mismatches = []
    for src, dst in copied:
        if src.stat().st_size != dst.stat().st_size or sha256(src) != sha256(dst):
            source_copy_hash_mismatches.append(dst.name)

    positions = [by_name[name]["r_Vo2To_vbs_true"] for name in mapping_names + query_names]
    ranges_m = [math.sqrt(sum(value * value for value in vector)) for vector in positions]
    summary = {
        "dataset": "SHIRT ROE1 synthetic pilot v1",
        "source": str(source_images),
        "window": {"first_frame": 1, "last_frame": 70, "count": 70},
        "selection": {
            "mapping_count": len(mapping_names),
            "query_count": len(query_names),
            "unused_buffer_count": len(buffer_names),
            "query_buffer_radius_frames": 1,
            "mapping_frames": mapping_frames,
            "query_frames": query_frames,
            "unused_buffer_frames": unused_buffer_frames,
        },
        "pose_coverage": {
            "window_cumulative_rotation_deg": sum(
                quaternion_angle_deg(
                    by_name[filename(frame)]["q_vbs2tango_true"],
                    by_name[filename(frame + 1)]["q_vbs2tango_true"],
                )
                for frame in range(1, 70)
            ),
            "mapping_temporal_gap_frames": {
                "min": min(mapping_temporal_gaps),
                "max": max(mapping_temporal_gaps),
            },
            "mapping_adjacent_attitude_gap_deg": {
                "min": min(mapping_angle_gaps),
                "max": max(mapping_angle_gaps),
            },
            "selected_translation_range_m": {"min": min(ranges_m), "max": max(ranges_m)},
            "query_nearest_mapping": query_nearest,
        },
        "verification": {
            "mapping_query_overlap": False,
            "all_selected_images_have_labels": True,
            "all_selected_files_exist": True,
            "source_copy_sha256_mismatches": source_copy_hash_mismatches,
        },
    }
    (output / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    make_contact_sheet(
        [mapping_dir / name for name in mapping_names],
        qa_dir / "mapping_contact_sheet.jpg",
        columns=8,
    )
    make_contact_sheet(
        [query_dir / name for name in query_names],
        qa_dir / "query_contact_sheet.jpg",
        columns=5,
    )

    min_query_gap = min(item["attitude_gap_deg"] for item in query_nearest)
    max_query_gap = max(item["attitude_gap_deg"] for item in query_nearest)
    readme = f"""# SHIRT ROE1 synthetic 小数据集 v1

## 用途

用于验证“航天器掩膜 → HLoc 特征匹配 → COLMAP/SfM 稀疏重建 → query 定位 → PnP 位姿估计”能否在小规模数据上跑通。

## 划分

- 源序列：`shirtv1/roe1/synthetic/images`
- 连续窗口：`img000001.jpg`–`img000070.jpg`
- Mapping：{len(mapping_names)} 张
- Query：{len(query_names)} 张
- 缓冲但不使用：{len(buffer_names)} 张
- Mapping 与 query 无交集；每张 query 的前后相邻帧均不进入 mapping。

之所以不在全部 2,371 帧中均匀抽样，是因为 ROE1 每帧约旋转 4.7°，全序列累计旋转约 31 圈。全局均匀抽样会造成相邻 mapping 视角跨度过大，并引入多圈后的近重复姿态。

## 视角检查

- 70 帧窗口累计旋转：{summary['pose_coverage']['window_cumulative_rotation_deg']:.3f}°
- Mapping 相邻时间间隔：{min(mapping_temporal_gaps)}–{max(mapping_temporal_gaps)} 帧
- Mapping 相邻姿态差：{min(mapping_angle_gaps):.3f}°–{max(mapping_angle_gaps):.3f}°
- Query 到最近 mapping 的姿态差：{min_query_gap:.3f}°–{max_query_gap:.3f}°
- 已选图像相对距离：{min(ranges_m):.3f}–{max(ranges_m):.3f} m

## 目录

```text
images/mapping/             40 张建图图像
images/query/               10 张查询图像
labels/mapping.json         mapping 真值标签
labels/query.json           query 真值标签
manifests/mapping.txt       HLoc 相对路径清单
manifests/query.txt         HLoc 相对路径清单
manifests/buffer_unused.txt 未使用缓冲帧
config/camera_shirt.json    原始 SHIRT 相机参数
config/colmap_camera.json   COLMAP FULL_OPENCV 参数
qa/*_contact_sheet.jpg      视角覆盖检查图
split_summary.json          划分指标与完整性验证
```

## 下一步

先为这 50 张图像生成航天器二值掩膜，再开始 HLoc/COLMAP 特征提取与稀疏重建。首轮基线只使用 synthetic 域，不做跨域定位。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
