# YOLO Mask 离线接入定位

## 做了什么

第 18 步读取同一组已保存的 RGB、轴向深度和采集时相机内外参。YOLO11n-seg 仅从 RGB 生成瓶子 Mask，再调用第 11 步原有的反投影和固定朝向模板匹配算法。仿真 Mask 单独运行相同算法作为对照，不参与 YOLO 目标选择或定位。没有启动机械臂、IK 或实时抓取。

第 11 步只把 Isaac 相关导入移入需要它们的函数，数值算法保持不变。第 18 步可以在训练 `.venv` 中运行，不需要给该环境安装 Isaac。

## 运行

在 `D:\qtproject\IssacSim Pj` 的 PowerShell 中：

```powershell
.\.venv\Scripts\python.exe perception\compare_pose.py
```

默认读取 `outputs/wrist_rgbd_01` 和 `outputs/yolo_train/multiview500/weights/best.pt`。可用 `--capture`、`--weights`、`--template` 指定其他输入。每次创建独立的输出目录，不覆盖原图、Mask 或历史定位报告。

首次或更换资产后，在 Isaac Script Editor 导出瓶身局部坐标系下的模板顶点：

```python
import runpy
from pathlib import Path
import omni.usd

root = Path(r"D:\qtproject\IssacSim Pj")
module = runpy.run_path(str(root / "perception/compare_pose.py"))
module["export_template"](
    omni.usd.get_context().get_stage(),
    root / "outputs/perception_templates/bottle014_body.npz",
)
```

此操作只读取局部几何，不修改场景，不把当前抓取结束后的世界位姿传给定位器。当前离线实验明确采用瓶体直立、旋转矩阵为单位阵的先验，只估计平移。

## 本次实测

结果目录：`outputs/yolo_pose_compare/20261119_123507`。

| 指标 | 仿真 Mask | YOLO Mask |
|---|---:|---:|
| 有效采样深度点 | 1513 | 1463 |
| XY 定位误差 | 1.511 mm | 1.519 mm |
| Z 定位误差 | +13.347 mm | +13.398 mm |
| 三维位置误差 | 13.433 mm | 13.484 mm |

YOLO 置信度 0.913，Mask IoU 0.962。两种输入导致的 XY 估计差为 0.039 mm，三维估计差为 0.064 mm。仿真 Mask 重算结果与原报告在浮点精度内一致。

脚本中的 5 mm 阈值只检查替换 Mask 引入的 XY 差异，是本步工程对照阈值，不是抓取安全性或绝对定位准确性的保证。

## 输出怎么看

- `report.json`：输入哈希、模型置信度、点数、残差、定位对照和评测误差。
- `comparison_overlay.png`：绿色为仿真 Mask/估计原点，红色为 YOLO Mask/估计原点。原点是瓶体参考系原点，不等于最终 TCP 抓取点。
- `yolo_mask.png` / `sim_mask.png`：原始分辨率的二值 Mask。
- `*_observed_world.ply`：经过 Mask 腐蚀、有效深度筛选和采样后的世界坐标点云。
- `*_fitted_template_world.ply`：模板应用估计平移后的世界坐标顶点。

## 当前边界与下一步

这是一个保存帧的离线替换实验，不能表示批量成功率或真实场景泛化。瓶型已知、朝向固定；现有模板只有瓶身，而 Mask 包含瓶盖，残差和高度偏差需要在后续几何定位优化中分析，不能简单归因于 YOLO。

旧 `camera_info.json` 中的 `bottle_body_translation_world_m_evaluation_only` 实际来自瓶子的父节点，本步不使用它。评测真值来自该次采集的 `bottle_pose_estimate.json` 中明确标记的瓶体原点。新采集接口应分别保存父节点与刚体的位姿，避免原点混淆。

当前抓取脚本使用估计 XY；抓取 Z 来自桌面高度加配置高度。因此 XY 替换对照通过，不等于已解决 Z 的约 13.4 mm 偏差。

下一步才是实时采集编号绑定 RGB-D、外部 YOLO 推理返回同编号 Mask、定位结果检查，然后执行一次抓取。
