# Isaac Sim 视觉抓取与合成数据 Demo

基于 Isaac Sim / PhysX、Franka Panda 和 RGB-D 手眼相机，实现水瓶视觉定位、物理抓取放置、自动评测，以及不执行抓取的合成数据采集。支持 ROS2 放置目标请求、阶段反馈和取消。

当前验证环境为 **Isaac Sim 6.1.0-rc.26 / PhysX**。主对象为 Lightwheel `bottle014`，YCB 芥末瓶作为隐藏回退资产。

## 功能链路

```text
手眼相机 RGB → YOLO 实例分割 Mask（可选仿真 Mask 基线）
    + 同帧深度与相机内外参
    → 已知网格模板平移估计 → Lula IK → 连续关节驱动
    → 瓶身中部渐进夹持 → 抬升 → 搬运 → 放置 → 松爪撤离
    → JSON 阶段诊断与任务结果

固定随机种子 + 瓶子位置/灯光扰动
    → 不执行抓取的自动采集 → RGB / Depth / Mask / YOLO 分割标签
```

- **物理交互**：动态刚体、碰撞代理、自定义加长夹指、渐进闭爪及有限预紧。
- **运动与评测**：IK 可达性预检、关节限速、TCP 跟踪监控、滑移/倾倒/超时判断、放置误差与失败阶段记录。
- **视觉定位**：YOLO 分割或仿真 Mask 基线、深度反投影、相机到世界变换、已知模板拟合；仍采用固定朝向先验。
- **数据采集**：可复现的随机参数、瓶身/瓶盖覆盖检查、边缘截断与过小目标过滤、逐样本元数据。
- **ROS2**：任务话题接口与状态反馈，可结合场景中的关节状态、TF、仿真时间发布进行 bag 录制。

## 快速运行

先按 [资产准备说明](docs/assets.md) 准备水瓶及纹理，打开根目录的 `Isaac_Desk_Arm_Bottle.usda`，暂停现有仿真并确认无其他控制任务。

在 **Isaac Sim Script Editor** 中运行，修改 `PROJECT` 为本机项目根目录：

```python
from pathlib import Path
import runpy

PROJECT = Path(r"D:\qtproject\IssacSim Pj")
entry = runpy.run_path(str(PROJECT / "apps/run_pick.py"))
task = entry["start"](
    weights=str(PROJECT / "outputs/yolo_train/multiview500/weights/best.pt"),
    place_xy_world_m=[0.18, 0.20],
    device="cpu",
)
```

脚本自动复位、采集当前图像、调用 YOLO、定位并执行抓放；任务结束后暂停。成功轮次只保留结果 JSON、RGB、YOLO Mask 和位姿估计；失败轮次保留中间文件用于排查。运行会在 `outputs/yolo_pick/` 下创建单轮目录，并默认使用 multiview500 权重；模型权重需本地准备。`omni.*`、PhysX、Lula 和 Replicator 来自 Isaac 环境，不能直接用普通系统 Python 启动这些编辑器脚本。

| 日常入口 | 用途 | 运行环境 |
| --- | --- | --- |
| `apps/run_pick.py` | YOLO 单次视觉抓放 | Isaac Script Editor |
| `apps/collect_dataset.py` | 自动采集随机化数据集，不抓取 | Isaac Script Editor |
| `apps/start_ros2_bridge.py` | 常驻 ROS2 接口，默认仿真 Mask | Isaac Script Editor |
| `apps/run_randomized_trial.py` | 单轮随机化抓取，默认仿真 Mask，保存 HDF5/JSON | Isaac Script Editor |
| `tool/dataset_studio/panel.py` | 双视口采集面板 | Isaac Script Editor |
| `perception/evaluate_model.py` | 已有 YOLO 模型验证 | 项目 `.venv` |
| `perception/compare_pose.py` | 保存帧的 YOLO/仿真 Mask 定位对照 | 项目 `.venv` |

目录职责、调用链和新旧入口对照见 [代码目录与运行指南](docs/code_layout.md)。原 `scripts/` 编号文件保留为兼容入口，修改业务代码请进入新目录。


## 已验证结果

| 实验 | 范围 | 结果 |
| --- | --- | --- |
| 视觉瓶身抓取与放置 | 当前场景单轮 | 成功；最大抬升 123.2 mm，最大滑移 6.5 mm，最终 XY 误差 11.2 mm |
| YOLO 分割驱动抓取与放置 | 当前场景完整成功样例；后续运行仍有撤离超时 | 成功样例最大抬升 119.9 mm，最大滑移 8.6 mm，最终 XY 误差 2.9 mm；见 [联调与验收记录](docs/yolo_live_pick.md) |
| 固定坐标瓶颈抓取 | 相同初始条件 10 轮，只抬升保持 | 10/10 成功，平均抬升 73.0 mm |
| 自动数据采集 | 54 次采集尝试 | 50 张合格样本，训练/验证 40/10 |

保留了 [原始示例报告](examples/reports)；视觉单轮成功与固定坐标 10 轮是不同实验。批量图像、HDF5、模型权重、缓存和 Python 虚拟环境由本地生成。

每轮视觉抓取结束后自动刷新离线报告 `outputs/evaluation_report.html`，展示成功率、误差折线、失败阶段和单轮视觉证据；也可运行 `python tools/build_evaluation_report.py` 手动重建。支持成功/失败筛选，统计口径见 [任务评测报告](docs/evaluation_report.md)。

`outputs/` 的目录用途与归档位置见 [输出目录说明](docs/outputs_layout.md)。

当前总质量约 560 g、摩擦和驱动参数为仿真假设；加长夹指与增大的开口为自定义仿真夹具。YOLO 接入操作见 [实时视觉抓放](docs/yolo_live_pick.md)，原始 Mask 对照见 [离线定位验证](docs/yolo_offline_pose.md)。未接入真实相机或真实机械臂，不将这些结果表述为 Sim2Real 部署完成。

## ROS2 任务输入

在 Isaac 中运行 `apps/start_ros2_bridge.py`，Ubuntu ROS2 终端与 Isaac 配置相同的 Domain ID 和可互通的 DDS 环境后：

```bash
ros2 topic echo /pick_status
```

另一个终端发送放置位置，单位米：

```bash
ros2 topic pub --once /pick_place_goal geometry_msgs/msg/PointStamped \
  "{header: {frame_id: 'world'}, point: {x: 0.18, y: 0.20, z: 0.0}}"
```

抓取位置由相机数据估计；请求只指定 world 坐标系的放置 XY。`/pick_status` 返回 JSON 状态和 `task_id`，`/pick_cancel` 接收包含对应 `task_id` 的 JSON 字符串。当前为话题接口，不是 ROS2 Action。

rosbag2 记录/回放的是 ROS 消息，当前没有将关节状态回放驱动到 Isaac 的播放器。回放历史目标话题可能触发新任务；查看记录时只选择所需状态话题。

## 数据与依赖

通过 `apps/collect_dataset.py` 和 `configs/dataset_randomization.json` 或 Dataset Studio 面板配置种子、数量、瓶子偏移、灯光方位角/仰角和强度，以及独立采集相机的方位角/俯视角/距离。左侧显示采集相机，右侧显示固定场景视角，原抓取手眼相机保持独立。输出 `images/`、`labels/`、`masks/`、`depth/`、`metadata/`、`samples.jsonl`、`data.yaml` 和可自动播放的 `gallery.html`。采集后不需要机械臂执行动作。操作与录屏方式见 [域随机化采集面板](docs/dataset_studio.md)。

视觉与采集使用 NumPy、Pillow、OpenCV；单轮随机化报告额外使用 h5py。请在与 Isaac 配套的 Python 环境中检查依赖。YOLO 训练使用独立环境，当前仓库不附带训练权重。
