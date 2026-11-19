# YOLO 实时采集与视觉抓放

## 入口

打开当前瓶子桌面场景，暂停或停止仿真。在 Isaac Script Editor 执行：

```python
import runpy
runpy.run_path(
    r"D:\qtproject\IssacSim Pj\scripts\19_yolo_visual_pick.py",
    run_name="__main__",
)
```

程序会 Stop 复位、拍摄一次 RGB-D、运行外部 YOLO，定位和 IK 通过后自动执行抓放，结束时暂停。默认放置到 world XY=(0.18, 0.20) m。

指定放置点或其他权重：

```python
import runpy
entry = runpy.run_path(r"D:\qtproject\IssacSim Pj\scripts\19_yolo_visual_pick.py")
task = entry["start"](place_xy_world_m=[0.18, 0.20], device="cpu")
```

依赖项目目录 `.venv` 中的 Ultralytics、PyTorch、OpenCV；默认权重为 `outputs/yolo_train/baseline50_run2/weights/best.pt`。模型文件未纳入 Git。

## 数据如何流动

1. **19：运行入口**。拒绝重复启动，建立独立运行目录，记录阶段、异常和最终完整抓取报告。
2. **12：实时采集和编排**。`mask_source='yolo'` 时只挂载 RGB 和 `distance_to_image_plane` 标注器，不挂载实例分割。采集时保存相机内外参和独立 capture ID。
3. **yolo_mask_inference.py：外部推理**。独立 `.venv` 进程读取 RGB，输出 `yolo_mask.png`、检测可视化和 JSON。默认 CPU，不将 PyTorch 加入 Isaac Python 环境。等待期间让出 Kit 更新循环。
4. **11：定位**。用 YOLO Mask 筛选深度，腐蚀边缘、过滤无效深度、采样，使用 USD 相机的行向量矩阵约定反投影到世界。已知瓶型模板和固定直立朝向先验参与平移估计。
5. **09：执行**。使用估计 XY 及桌面+11 cm 的配置抓取高度，通过原有 IK/状态机完成抓取、抬升、搬运、放置、撤离及稳定检查。

此入口不使用仿真 Mask 选择目标，也不在识别失败时回退到仿真真值。已知瓶子网格、固定朝向和配置抓取高度仍是本 Demo 的先验。

执行器仍读取仿真物体状态做夹持相对位姿、搬运目标修正、滑移/支撑与结果检查。这不是完全摆脱仿真状态反馈的真实机器人部署方案；本次替换的是视觉目标定位输入。

## 错误处理

- 请求与结果的 capture ID、RGB SHA256、Mask SHA256、图像尺寸都必须匹配。
- 置信度阈值默认 0.5；当前只接受恰好一个瓶子，多瓶子尚未实现目标选择。
- Mask 至少 1200 像素，不能触及图像边界；有效采样深度点至少 50 个。
- 推理超时 90 秒终止子进程；异常详细信息保存到 `yolo_worker.log`。
- 推理期间检查场景和相机是否改变；改变时要求重新采集。
- 模板匹配中位残差不超过 15 mm、P90 不超过 60 mm；目标 XY 必须位于桌面工作区域；IK 检查失败不启动抓取。这些是初版工程阈值，不等于安全认证或全场景可靠性。
- 仿真抓放阶段仍使用原有超时、取消、滑移、放置及撤离检查。

## 保存在哪里

每次输出到 `outputs/yolo_pick/<时间_编号>/`，摘要入口是 `outputs/yolo_pick_latest.json`。

| 文件 | 内容 |
|---|---|
| rgb.png / depth_m.npy / depth_noisy_m.npy | 同次采集的图像、原始深度和定位使用的深度副本 |
| camera_info.json | capture ID、内参、采集时外参；父节点与瓶体真值分别命名，仅用于评测 |
| yolo_request.json / yolo_result.json | 编号与哈希、置信度、框、模型版本、推理耗时 |
| yolo_mask.png / yolo_prediction.png | 本次预测 Mask 与检测展示图 |
| bottle_pose_estimate.json | 估计位置、残差、真值误差及 IK 预检查 |
| result.json | 阶段时间线与最终完整抓放结果 |

旧入口 12 默认仍为 `mask_source='sim'`，ROS2 适配器目前也保持原有默认；本次新增入口 19 用于 YOLO 验收。后续切换 ROS2 时应显式传入 `mask_source='yolo'`。

## 当前边界

这是“停止观察→单帧识别定位→开环抓放执行”，尚未实现运动中的持续视觉伺服。姿态未从图像估计。当前几何定位约 13 mm 的 Z 偏差不会因替换 Mask 自动消失，执行器只使用其 XY；瓶身/瓶盖模板一致性和高度定位需后续单独优化。

## 联调中发现的控制问题

首次 YOLO 完整运行 `20261119_130528_5b8bbc4d` 在 Lower 阶段因滑移 15.08 mm 超过 15 mm 阈值而失败。运行报告保留在对应目录。该轮置信度 0.916，视觉 XY 误差约 1.52 mm，已完成抬升与水平搬运。

日志显示 Lower 阶段按每步物体真值更新 XY 修正，接触晃动时目标 X 从约 0.1795 m 漂移到 0.1911 m，伴随倾斜与滑移增长。控制器改为搬运结束时计算一次 XY 修正，下降时保持该目标，避免持续追踪晃动的物体。验收阈值未放宽，是否改善由后续完整运行结果判断。

第二轮 `20261119_131132_a57b8116` 完成下降、松爪、撤离，最大滑移 6.35 mm，但最终 XY 误差 20.30 mm，Settle 判失败。下降期间 XY 目标变化为 0；松爪时瓶体约 7° 倾斜，随后恢复直立，瓶体原点横移约 2 cm。

进一步修改为支撑中心对齐：在搬运结束时计算当前 `body_position + body_rotation × local_base_center`，与目标直立位姿的支撑中心比较，得到一次性的 XY 修正。下降期间仍固定目标。松爪前按支撑中心检查 5 mm 门限，最终仍按瓶体原点检查 15 mm 放置门限。计算使用执行器的仿真状态反馈，不能表述为纯视觉在线闭环控制。新增瓶体四元数诊断，便于追踪倾斜恢复。

## 完整链路验收结果（2026-11-19）

第三轮 `20261119_131851_aa31039f` 最终 **Succeeded / Settle**，完成实时采集、外部 CPU YOLO、深度定位、IK、夹持、搬运、松爪、撤离和直立稳定检查。

| 指标 | 结果 |
|---|---:|
| 抓放仿真耗时 | 39.49 s |
| 含采集/推理的总墙钟耗时 | 348.06 s |
| 最大抬升 | 119.88 mm |
| 最大夹持/搬运滑移 | 8.60 mm |
| 最终 XY 放置误差 | 2.91 mm |
| 下降阶段 XY 目标变化 | 0 mm |

全部数据保存在 `outputs/yolo_pick/20261119_131851_aa31039f/`。目录没有仿真 `bottle_mask.png`，定位报告的输入为 `yolo_mask.png`。相机、推理结果和定位报告的 capture ID 一致。原有 15 mm 滑移和最终放置误差检查没有放宽。

这是修改后的单轮通过，不是批量成功率。前两次失败运行不删除，也不混入修改后版本的成功率统计。

后续运行 `20261119_132911_4956fff2` 在 `RetreatClear` 阶段超时，结束时 XY 放置误差约 5.13 mm，但整轮必须记为未通过。因此当前已有完整成功样例，重复运行的撤离稳定性仍需排查。仓库同时保存成功报告和各轮结果摘要，不能据此宣称稳定批量成功。

可随仓库查看的记录：[完整成功报告](../examples/reports/yolo_pick_success.json)、[各轮联调摘要](../examples/reports/yolo_pick_runs_summary.json)、[后续撤离超时报告](../examples/reports/yolo_pick_retreat_timeout.json)。

已验证的异常分支：采集编号不一致、RGB 哈希不一致、未检出瓶子都会被拒绝。推理进程异常/超时、深度与残差门限也已实现，但未对每一种异常做故障注入测试。
