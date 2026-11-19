# Isaac Sim 视觉抓取与合成数据 Demo

基于 Isaac Sim / PhysX、Franka Panda 和 RGB-D 手眼相机，实现水瓶视觉定位、物理抓取放置、自动评测，以及不执行抓取的合成数据采集。支持 ROS2 放置目标请求、阶段反馈和取消。

当前验证环境为 **Isaac Sim 6.1.0-rc.26 / PhysX**。主对象为 Lightwheel `bottle014`，YCB 芥末瓶作为隐藏回退资产。

## 功能链路

```text
手眼相机 RGB + 深度 + 实例 Mask
    → 已知网格模板平移估计 → Lula IK → 连续关节驱动
    → 瓶身中部渐进夹持 → 抬升 → 搬运 → 放置 → 松爪撤离
    → JSON 阶段诊断与任务结果

固定随机种子 + 瓶子位置/灯光扰动
    → 不执行抓取的自动采集 → RGB / Depth / Mask / YOLO 分割标签
```

- **物理交互**：动态刚体、碰撞代理、自定义加长夹指、渐进闭爪及有限预紧。
- **运动与评测**：IK 可达性预检、关节限速、TCP 跟踪监控、滑移/倾倒/超时判断、放置误差与失败阶段记录。
- **视觉基线**：深度反投影、相机到世界变换、已知模板拟合；使用仿真真值 Mask 和固定朝向先验。
- **数据采集**：可复现的随机参数、瓶身/瓶盖覆盖检查、边缘截断与过小目标过滤、逐样本元数据。
- **ROS2**：任务话题接口与状态反馈，可结合场景中的关节状态、TF、仿真时间发布进行 bag 录制。

## 快速运行

先按 [资产准备说明](docs/assets.md) 准备水瓶及纹理，打开根目录的 `Isaac_Desk_Arm_Bottle.usda`，暂停现有仿真并确认无其他控制任务。

在 **Isaac Sim Script Editor** 中运行，修改 `PROJECT` 为本机项目根目录：

```python
from pathlib import Path
import runpy

PROJECT = Path(r"D:\qtproject\IssacSim Pj")
runpy.run_path(str(PROJECT / "scripts/12_visual_bottle_pick.py"), run_name="__main__")
```

脚本自动复位、采集当前图像、定位并执行抓放；任务结束后暂停。结果位于 `outputs/lesson09_latest.json`，入口摘要为 `outputs/visual_pick_runner_latest.json`。`omni.*`、PhysX、Lula 和 Replicator 来自 Isaac 环境，不能直接用普通系统 Python 启动这些编辑器脚本。

| 入口 | 用途 |
| --- | --- |
| `scripts/09_pick_and_place_bottle.py` | 已知坐标或外部估计坐标的瓶身抓放控制器 |
| `scripts/10_ros2_pick_task.py` | ROS2 常驻任务接口，调用视觉抓放 |
| `scripts/11_register_bottle_translation.py` | 从已采集 RGB-D/Mask 估计瓶子平移 |
| `scripts/12_visual_bottle_pick.py` | 单次完整视觉抓放 |
| `scripts/13_single_domain_randomization.py` | 单轮位置/灯光/深度噪声扰动，保存 HDF5/JSON |
| `scripts/14_generate_bottle_dataset.py` | 不抓取，自动生成分割数据集 |
| `scripts/15_fixed_pose_waterbottle_grasp.py` | 固定坐标瓶颈夹持、抬升保持基线 |
| `scripts/16_repeat_fixed_pose_waterbottle_grasp.py` | 固定抓取重复测试及汇总 |
| `scripts/franka_visual_compat.py` | Franka 连杆与夹爪实例几何显示兼容 |

第 1～8 课为历史学习与资产导入脚本，包含不同对象/场景假设，不应直接套用到当前水瓶。学习材料见 [学习拼图](学习拼图.md) 与 [历史课程记录](docs/lesson_history.md)。当前版本的参数和结果以 [验证记录](docs/current_validation.md) 为准。

## 已验证结果

| 实验 | 范围 | 结果 |
| --- | --- | --- |
| 视觉瓶身抓取与放置 | 当前场景单轮 | 成功；最大抬升 123.2 mm，最大滑移 6.5 mm，最终 XY 误差 11.2 mm |
| 固定坐标瓶颈抓取 | 相同初始条件 10 轮，只抬升保持 | 10/10 成功，平均抬升 73.0 mm |
| 自动数据采集 | 54 次采集尝试 | 50 张合格样本，训练/验证 40/10 |

保留了 [原始示例报告](examples/reports)；视觉单轮成功与固定坐标 10 轮是不同实验。批量图像、HDF5、模型权重、缓存和 Python 虚拟环境由本地生成。

当前总质量约 560 g、摩擦和驱动参数为仿真假设；加长夹指与增大的开口为自定义仿真夹具。当前未接入 YOLO 推理、真实相机或真实机械臂，不将这些结果表述为 Sim2Real 部署完成。

## ROS2 任务输入

在 Isaac 中运行第 10 课入口，Ubuntu ROS2 终端与 Isaac 配置相同的 Domain ID 和可互通的 DDS 环境后：

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

第 14 课默认采集 50 个样本，可在脚本顶部调整种子、数量、位置与灯光范围。输出 `images/`、`labels/`、`masks/`、`depth/`、`metadata/`、`samples.jsonl` 和 `data.yaml`。采集后不需要机械臂执行动作。

视觉与采集使用 NumPy、Pillow、OpenCV；第 13 课额外使用 h5py。请在与 Isaac 配套的 Python 环境中检查依赖。YOLO 训练使用独立环境，当前仓库不附带训练权重。
