# 代码结构与运行入口

项目按任务入口、仿真模块、独立感知进程和诊断工具组织。任务脚本在 Isaac Sim Script Editor 中执行，YOLO 推理与离线验证使用项目独立的 `.venv`。

## 目录职责

| 目录 | 内容 |
|---|---|
| `apps/` | 单次 YOLO 抓放、数据采集、ROS2 接口、单轮随机化入口 |
| `src/bottle_sim/` | 机器人初始化、运动学、抓放状态机、视觉定位与采集模块 |
| `perception/` | 外部 YOLO 推理、模型验证、定位对照 |
| `tool/dataset_studio/` | 双视口数据采集面板 |
| `tools/build_evaluation_report.py` | 生成离线 HTML 任务评测报告 |
| `tools/diagnostics/` | 固定坐标单轮/重复抓取诊断 |
| `scripts/` | 16 个旧功能名称的轻量兼容入口 |
| `configs/` | 数据采集域随机化配置 |
| `examples/reports/` | 已保存的示例评测报告 |
| `assets/` | 本地准备的场景资产，获取方式见资产文档 |
| `outputs/` | 本地生成的图片、模型与报告，不纳入 Git |

教学脚本不属于仓库运行依赖。实际实现只有一份；旧编号入口调用新位置，维护时请修改新位置。

## 单次 YOLO 抓放

打开项目根目录场景，暂停仿真并确认无其他任务。在 Isaac Script Editor 中执行，按本机目录修改 `PROJECT`：

```python
from pathlib import Path
import runpy

PROJECT = Path(r"D:\qtproject\IssacSim")
entry = runpy.run_path(str(PROJECT / "apps/run_pick.py"))
task = entry["start"](
    place_xy_world_m=[0.18, 0.20],
    weights=str(PROJECT / "outputs/yolo_train/multiview500/weights/best.pt"),
    device="cpu",
)
```

模型权重需本地准备；默认使用 multiview500。每轮输出为 `outputs/yolo_pick/<运行编号>/`，结束后自动刷新 `outputs/evaluation_report.html`。成功轮次保留结果 JSON、RGB、YOLO Mask 和位姿估计；失败轮次保留中间诊断文件。

## 其他入口

| 入口 | 用途 |
|---|---|
| `apps/collect_dataset.py` | 不抓取，自动采集随机化数据集 |
| `apps/start_ros2_bridge.py` | 常驻 ROS2 任务接口，当前默认仿真 Mask |
| `apps/run_randomized_trial.py` | 单轮随机化抓取，当前默认仿真 Mask，保存 JSON/HDF5 |
| `tool/dataset_studio/panel.py` | 交互配置采集参数、显示相机与场景视图 |

这些入口可用 `runpy.run_path(完整路径, run_name="__main__")` 执行。单轮随机化要求 Stop 状态。当前没有统一的多轮 YOLO 随机抓取入口；固定坐标重复诊断与随机化图片采集是不同流程。

## 调用关系

```text
apps/run_pick.py
  └─ src/bottle_sim/visual_pick.py     RGB-D 采集与流程编排
       ├─ perception/yolo_worker.py   外部进程推理与结果校验
       ├─ pose_estimation.py          深度反投影、模板拟合、IK 检查
       └─ pick_task.py                物理配置、抓放、指标与报告
            └─ motion.py             运动学、TCP 与几何辅助
                 └─ robot.py         初始化与关节状态读取

apps/collect_dataset.py
  └─ dataset.py                      采集、自动标注、质量筛选
       └─ randomization.py           配置、随机采样、图库生成
tool/dataset_studio/panel.py           调用 dataset.py
```

修改抓放参数与阶段判断看 `pick_task.py`；修改采集与视觉流程看 `visual_pick.py`；修改坐标变换与定位看 `pose_estimation.py`；修改采集范围看 `configs/dataset_randomization.json`。

`ros2_bridge.py` 处理任务消息，`randomized_trial.py` 处理单轮随机化与 HDF5，`visual_compat.py` 处理机械臂实例几何显示兼容。

## 加载与兼容

新旧功能入口对照见 [script_migration.json](script_migration.json)。部分控制器仍按任务动态加载，保留独立的参数命名空间；没有直接改为共享缓存的模块实例。后续可在任务参数改为显式配置对象后再统一普通 import。

`pick_task.py` 仍包含物理配置、控制与报告，当前整理以目录职责和依赖路径为主。不要只复制 `scripts/`，需保留完整项目结构。

## 验证

普通 Python 可运行：

```bash
python tools/check_layout.py
```

检查覆盖语法、兼容入口、项目根目录和内部 Python 文件引用。整理时还检查了 Isaac 中的新旧模块加载、随机化参数一致性，以及外部 YOLO 在已有图片上的推理结果一致性。这些检查不代替完整抓放回归；抓放结果范围见 [验证记录](current_validation.md)。
