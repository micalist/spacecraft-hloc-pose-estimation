# Spacecraft HLoc Pose Estimation

将 [Hierarchical-Localization (HLoc)](https://github.com/cvg/Hierarchical-Localization)
迁移到无 CAD、非合作航天器图像上的实验代码。项目把航天器视为一个可重建的刚体场景，
先用 Mapping 图像建立稀疏三维模型，再通过 2D–3D 匹配与 PnP 求解 Query 相机相对
航天器模型的六自由度位姿。

> 当前状态：已完成 **SHIRT/ROE1 synthetic 数据上的端到端可行性验证**，包括数据划分、
> 掩膜特征过滤、SfM、Query 定位、位姿误差评估和 Mapping 规模实验。它是研究原型，
> 还不是经过多数据集、真实相机和在轨条件验证的通用系统。

## 方法概览

```text
Mapping images
  -> ALIKED 特征 + 航天器 ROI 掩膜过滤
  -> 图像匹配与几何验证
  -> COLMAP 增量 SfM
  -> 航天器坐标系中的稀疏三维点

Query image
  -> 2D 特征与 Mapping/3D 轨迹关联
  -> PnP + RANSAC
  -> Query 相机相对航天器模型的 R, t
```

不需要输入 CAD 模型；PnP 所需三维点由 Mapping 图像的 SfM 重建产生。这里估计的是
相机与重建模型坐标系的相对位姿。若要输出严格的航天器本体坐标系位姿，还需要用真值、
标志点或其他先验完成模型坐标系对齐。

## 已完成实验

固定 10 张 Query，使用 40、100、200、400 张嵌套 Mapping 集合得到以下单次实验结果：

| Mapping | 注册图像 | 三维点 | 可靠 Query | 旋转误差中位数 | 平移误差中位数 |
|---:|---:|---:|---:|---:|---:|
| 40 | 40/40 | 2,843 | 10/10 | 2.882° | 0.274 m |
| 100 | 100/100 | 7,317 | 10/10 | 2.152° | 0.188 m |
| 200 | 158/200 | 10,933 | 10/10 | 3.717° | 3.055 m |
| 400 | 158/400 | 10,839 | 10/10 | 0.769° | 0.156 m |

结果说明 100 张是本轮同时改善完整注册和定位精度的规模点；200/400 张均在
`img000189.jpg` 附近停止增量注册。400 张的较低误差不能直接解释为“图像越多越准”，
因为当前结果仅有一次随机种子运行，且固定 Query 主要覆盖早期轨迹。完整分析见
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 仓库结构

```text
.
├─ scripts/
│  ├─ project_paths.py                         跨机器路径配置
│  ├─ check_environment.py                     环境与目录自检
│  ├─ prepare_roe1_pilot.py                    生成 40 Mapping + 10 Query 小数据集
│  ├─ run_hloc_pilot.py                        HLoc 全流程入口
│  ├─ diagnose_map40.py                        40 张重建分裂诊断
│  ├─ run_experiment5_model_unification.py     子模型统一与定位对照
│  └─ run_experiment6_mapping_scaling.py       40/100/200/400 规模实验
├─ config/env.example.ps1                      Windows 环境变量示例
├─ docs/EXPERIMENTS.md                         实验记录与限制
├─ requirements.txt
└─ .gitignore
```

数据集、模型权重、第三方仓库、虚拟环境、COLMAP 数据库和完整实验产物体积较大，
不会提交到本仓库。

## 安装

建议使用 Python 3.10–3.11。CUDA 不是必需的，但会显著加速特征提取和匹配。

```powershell
git clone https://github.com/micalist/spacecraft-hloc-pose-estimation.git
cd spacecraft-hloc-pose-estimation

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

git clone https://github.com/cvg/Hierarchical-Localization.git
git clone https://github.com/cvg/LightGlue.git
pip install -e .\LightGlue
pip install -e .\Hierarchical-Localization
```

如果 PyTorch 的默认安装不匹配本机 CUDA，请先按
[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装合适版本，再执行其余依赖安装。

## 数据准备

本仓库不再分发 SHIRT 数据。将数据放成以下结构，或通过环境变量指定实际位置：

```text
shirtv1/
├─ camera.json
└─ roe1/
   ├─ roe1.json
   └─ synthetic/images/
      ├─ img000001.jpg
      └─ ...
```

默认约定 HLoc、LightGlue、`shirtv1` 和实验输出均位于仓库根目录。路径也可覆盖：

```powershell
Copy-Item config\env.example.ps1 config\env.local.ps1
# 按需编辑 env.local.ps1，然后加载：
. .\config\env.local.ps1
```

运行自检：

```powershell
python scripts\check_environment.py
```

## 快速复现

生成确定性的 ROE1 小数据集：

```powershell
python scripts\prepare_roe1_pilot.py
```

逐阶段运行基础流程，便于定位失败步骤：

```powershell
python scripts\run_hloc_pilot.py --stage prepare
python scripts\run_hloc_pilot.py --stage extract
python scripts\run_hloc_pilot.py --stage match
python scripts\run_hloc_pilot.py --stage reconstruct
python scripts\run_hloc_pilot.py --stage localize
python scripts\run_hloc_pilot.py --stage evaluate
```

也可以一次运行：

```powershell
python scripts\run_hloc_pilot.py --stage all
```

后续实验入口：

```powershell
# 只读诊断 40 张 Mapping 的模型分裂
python scripts\diagnose_map40.py

# 合并保留子模型并使用同一批 Query 对照
python scripts\run_experiment5_model_unification.py

# Mapping 规模扩展；完整运行耗时和显存需求较高
python scripts\run_experiment6_mapping_scaling.py --stage all
```

所有脚本默认只在自己的实验输出目录中写入结果。原始 SHIRT 数据被视为只读输入，规模实验
还会在运行前后校验源文件聚合 SHA-256。

## 主要限制

- 目前只验证了 SHIRT ROE1 synthetic 的单条序列。
- 100–400 张实验是单随机种子，不构成统计显著性结论。
- 固定 Query 集集中于序列前段，尚不能充分评估后续视角覆盖。
- 187–189 帧附近存在 ROI 与 2D–3D 轨迹支持不足问题。
- 合成数据相机内参来自数据集；迁移到真实相机时必须先完成相机标定。

## 致谢

本项目基于 HLoc、COLMAP/pycolmap、ALIKED 和 LightGlue 生态完成。请在学术使用时同时遵循
这些上游项目与 SHIRT 数据集的许可及引用要求。
