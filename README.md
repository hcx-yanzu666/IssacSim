# Isaac Sim 机械臂抓取 · 教学练习

当前环境：Isaac Sim 6.1 / PhysX。第1～7课最初在5.1验证，保留原方块课程；第9课适配现有YCB瓶子。

学习主文档：[学习拼图.md](学习拼图.md) —— 每一步的目标、输入输出、关键 API、验收和小实验。后续按这份文档逐块推进。

当前仓库包含瓶子抓放场景 `Isaac_Desk_Arm_Bottle.usda`。场景引用 NVIDIA Nucleus 上的 Franka Panda 与 YCB mustard bottle USD；首次打开需要可访问对应 Isaac 官方资产 URL。原始方块场景未随仓库提交。`scripts/` 与 `学习拼图.md` 提供练习代码和教学记录。

## 学习路线 / TODO

- [x] 第 1 课：读取当前 Stage，找到机械臂、方块、桌面，检查尺寸与关节配置。
- [x] 第 2 课功能：初始化运行时 Articulation，读取 DOF 名称与实际关节位置；已通过 MCP 运行验证，待完成你的观察实验。
- [x] 第 3 课：空夹爪开合与双指联动（分步测试通过）。
- [x] 机械臂准备姿态与逐步目标运动：由第四、五课实现。
- [x] 第 4 课：Lula IK验证 + RMPflow运动生成，末端到方块上方悬停。
- [x] 第 5 课第一段：到上方、下降、夹紧、抬起并保持，已完成物理抓取测试。
- [x] 第 6 课：搬运到B点、下降、松开、撤离和放置验证，分步测试及自动运行入口均已通过。
- [x] 单轮运行：Stop恢复起点、阶段超时、抓取/放置判断和JSON结果记录。
- [x] 第 7 课：固定配置批量重跑、独立报告和成功率汇总。
- [x] 单轮域随机化入口：固定 seed 随机化瓶子 XY、灯光和深度噪声，保存 RGB-D、mask、HDF5 与结果 JSON。
- [ ] 自动数据集采集入口已实现，待 Isaac 场景运行验证：不执行抓取，批量随机化瓶子 XY 和灯光，生成 YOLO 分割标签及 RGB、Mask、Depth、逐样本元数据。
- [ ] 后续工程化：参数组合实验和不同物体对比。
- [ ] 后续：刚体水瓶 → 软体抓取 → RGB-D 数据生成 → 视觉定位与抓取。

## 现在运行什么

首次学习从 `scripts/01_inspect_scene.py` 开始；瓶子任务由第9课执行抓放、第12课接入RGB-D视觉、第10课提供ROS2放置目标接口。

在 Isaac Sim 的 Script Editor 中执行下面两行即可；也可以由 MCP 执行：

```python
import runpy
runpy.run_path(r"D:\qtproject\IssacSim\scripts\01_inspect_scene.py", run_name="__main__")
```

不要直接用系统 Python 执行：`omni.usd` 等模块来自已启动的 Isaac Sim 运行环境。
编辑器已经启动应用，所以这里也不再创建 `SimulationApp`。

## 第9课：已知瓶子位置调试入口

`09_pick_and_place_bottle.py` 的 `start` 支持可选的瓶子 Body 世界坐标，单位米：

```python
import runpy
lesson = runpy.run_path(r"D:\qtproject\IssacSim\scripts\09_pick_and_place_bottle.py")
lesson['start'](bottle_position_world_m=[0.18, -0.20, 0.8468])
```

先暂停当前仿真并确认没有其他控制任务。默认调用 `lesson['start']()` 仍读取场景瓶子位姿。外部参数用于计算抓取参考，不会移动瓶子；本课按桌面高度抓瓶身，因此仅使用输入的 X/Y，Z 会记录在报告中，但不决定 TCP 高度。输入必须对应实际瓶子 Body 原点而非视觉中心；错误的参考可能导致抓空或失败。实际瓶子位姿仍用于抓取与放置评测。

该接口用于离线调试已知瓶子位置。ROS2抓放链路不使用它作为抓取目标；第12课通过RGB-D估计瓶子位置后再调用第9课。

## ROS2：视觉抓瓶并放到指定位置

当前 Demo 使用普通 ROS2 话题完成任务请求与状态反馈，不额外引入自定义 Action 消息包。Isaac 适配器常驻运行，同一时间只执行一个任务；任务结束后会回到等待状态。

请求话题 `/pick_place_goal` 使用 `geometry_msgs/PointStamped`：`header.frame_id=world`，`point.x/y` 是放置位置（米），`point.z=0`。瓶子抓取位置来自 RGB-D 估计；放置高度由桌面和瓶底几何计算。状态话题 `/pick_status` 使用 JSON 字符串，报告接受、采图、定位、IK、抓放各阶段及最终结果；`/pick_cancel` 可请求取消当前任务。

确保 Isaac 时间线处于 Stop，然后在 Script Editor 运行：

```python
import runpy
runpy.run_path(r"D:\qtproject\IssacSim\scripts\10_ros2_pick_task.py", run_name="__main__")
```

在 Ubuntu 终端先监听状态，再发送目标：

```bash
ros2 topic echo /pick_status
```

另开终端发送放置目标：

```bash
ros2 topic pub --once /pick_place_goal geometry_msgs/msg/PointStamped \
  "{header: {frame_id: 'world'}, point: {x: 0.18, y: 0.20, z: 0.0}}"
```

状态消息带 `task_id`，便于关联同一轮任务。已有任务运行时新请求会返回 `rejected`；取消命令通过 `/pick_cancel` 发送 JSON `{"task_id":"<当前任务ID>"}`，Isaac 在下一个 Kit 更新点停止状态机并保存 `Cancelled` 报告。详细抓放数据保存在 `outputs/lesson09_latest.json`。

## 单轮域随机化

当前单轮入口 `scripts/13_single_domain_randomization.py` 每次使用脚本中的 `MASTER_SEED`，随机改变瓶子 XY（各±1.5cm）、环境灯强度、桌面灯强度和深度噪声标准差（0～3mm）。瓶子朝向固定。停止时间线后，脚本通过 USD 会话层应用场景扰动，然后复用第12课采 RGB-D/Mask 和抓放流程；原始深度保留在 `depth_m.npy`，带噪副本写入 `depth_noisy_m.npy` 并供第11课估计器使用。

在 Isaac Script Editor 运行：

```python
import runpy
runpy.run_path(r"D:\qtproject\IssacSim\scripts\13_single_domain_randomization.py", run_name="__main__")
```

结果写到 `outputs/domain_randomization/seed_<seed>_<时间>/`：RGB、Mask、原始/带噪深度、相机与mask元数据、估计结果、单轮结果 JSON 和 `trial.h5`。最近一次摘要另写入 `single_trial_latest.json`。这只是单轮链路验证；同 seed 的随机参数/噪声可复现，渲染和 PhysX 抓放不承诺逐位相同。当前真值 mask 与已知网格模板仍是仿真定位基线，并不代表 YOLO 的识别鲁棒性。

## 自动采集瓶体分割数据集

`scripts/14_generate_bottle_dataset.py` 只采集数据，不启动抓取任务。它按固定 master seed 随机化瓶子 XY 和两盏灯的强度，通过手眼相机采集 RGB、实例分割真值 Mask 与深度，并从 Mask 生成 YOLO segmentation 多边形标签。瓶子朝向保持固定；当前目标类为一个指定的瓶体资产。

运行前打开瓶子场景，确认场景 Prim 路径与脚本顶部配置一致，将机械臂停在瓶子位于相机画面中央的观察姿态，并暂停时间线（Stop 或 Pause 均可）。脚本会拒收 Mask 过小、目标框过小或碰到画面边缘的帧。在 Isaac Sim Script Editor 运行：

```python
import runpy
runpy.run_path(r"D:\qtproject\IssacSim\scripts\14_generate_bottle_dataset.py", run_name="__main__")
```

采集数量、训练集比例、随机范围和 seed 可在脚本顶部修改。数据写入 `outputs/bottle_dataset/batch_<seed>_<时间>/`，包含 `images/`、`labels/`、`masks/`、`depth/`、逐样本 `metadata/`、`samples.jsonl`、`summary.json` 和 YOLO `data.yaml`。脚本会跳过空 Mask/无效轮廓，并在结束时恢复瓶子位置与灯光强度。数据集使用的是仿真真值分割标注；它可用于训练和验证管线，不等于模型已经具备真实场景泛化能力。生成 YOLO 多边形标签需要 Isaac Python 环境可导入 `cv2`。

注意：`scripts/14_generate_bottle_dataset.py` 已实现数据采集与 YOLO 分割标签导出，但尚未在当前已保存场景中完成一批合格样本验收。场景目前仍使用 YCB 黄芥末瓶；Lightwheel 水瓶资产尚未集成，也未随仓库发布。

## rosbag2 命令行回放

在 Ubuntu 的两个终端中执行 `source /opt/ros/humble/setup.bash`，并设置与录制时相同的 `ROS_DOMAIN_ID` 和 `RMW_IMPLEMENTATION`。将下面的 `~/bags/isaac_pick_01` 替换为实际 bag 目录：

```bash
ros2 bag info ~/bags/isaac_pick_01
```

先在终端 A 监听任务状态，回放从开始到结束时观察状态变化：

```bash
ros2 topic echo /pick_status
```

终端 B 只回放任务状态和关节状态，避免旧目标触发新的抓取任务：

```bash
ros2 bag play ~/bags/isaac_pick_01 --topics /pick_status /joint_states
```

如需查看关节数据，可额外在终端 C 运行 `ros2 topic echo /joint_states --field position`。回放前确认 Isaac 中的 `/joint_states` 发布已停止，避免录制数据与当前场景数据混在一起；也不要同时发布多个 `/clock` 或同一组 TF。完成后用 Ctrl+C 停止监听。回放的是记录的 ROS 消息，当前不会驱动 Isaac 机械臂重演；bag 若未录制瓶子位姿也无法还原瓶子轨迹。

## 第一课抓住这条关系

`Stage（整个场景）→ Prim（某个对象）→ Schema/API（对象具备的能力）`

类似 UE 中“世界 → Actor → 组件能力”的理解方式，但 USD 的 Prim 并不等同于 UE Actor。
一个方块只有几何外观时不会自动成为动力学物体；CollisionAPI 提供碰撞，RigidBodyAPI 提供刚体能力。
机械臂的 Articulation 将连杆和关节组织成可由物理引擎求解的关节系统。

USD 中读到的关节限制和 drive 目标属于场景配置；物理运行时的实际关节位置需要通过 Articulation 读取。
场景中的关节遍历顺序不等于控制数组的 DOF 顺序，后面必须按 DOF 名称建立索引。

## 第二课的文件

现在运行 `scripts/02_read_joint_state.py`，已实现运行时关节状态读取。
旧文件 `02_joint_control_todo.py` 仅保留为后续控制流程参考，不是当前入口。
新脚本要求时间线停止或暂停；首次物理初始化可能执行预热步，不发送运动目标、不保存 USD。

```python
import runpy
runpy.run_path(r"D:\qtproject\IssacSim\scripts\02_read_joint_state.py", run_name="__main__")
```

## 新教学方式

每课四步：我实现一个可运行的小功能 → 讲 2～3 个关键 API → 你改一个参数并预测结果 → 根据输出讨论原因。
不用从空文件手写，也不要求背导入路径；重点是解释输入、输出、运行条件和异常。

第 2 课实验：

1. 运行脚本，观察 `panda_joint2` 的数组下标、弧度及角度。
2. 将开头 `WATCH_JOINT` 改为 `"panda_finger_joint1"`。
3. 先预测单位变化，再运行；它读取的是单个手指关节位移，而不是整个夹爪开口。
4. 将修改前后的“重点观察”输出发给我，然后进入单关节控制。

说明纠正：5.1 的 `SingleArticulation` 从 `isaacsim.core.prims` 导入。
本课使用 `SimulationManager.initialize_physics()`；后续任务生命周期再讲 World 和异步 reset。
之前聊天里的固定 9 元素动作数组仅为示意，不能直接照抄执行：必须核对实际 DOF 顺序和夹爪联动。

本次实测：9 个运行时 DOF；`panda_joint2` 位于下标 1，约 -0.568513 rad（-32.57°）。
第二课确认第二个手指存在 mimic 配置，引用第一个手指关节；动态验证见第三课。

## 第三课：让夹爪真正动起来

运行 `scripts/03_gripper_motion.py`，自动张开、闭合、再张开，各等待 3 秒仿真时间，最后暂停。
运行前先暂停现有任务；运行中点 Pause/Stop 可结束实验。

```python
import runpy
runpy.run_path(r"D:\qtproject\IssacSim\scripts\03_gripper_motion.py", run_name="__main__")
```

第二课的 `WATCH_JOINT` 只选“看谁”，不会移动机器人。
第三课的 `OPEN_POSITION_M` 才是驱动目标：默认 0.03 m，每根手指 30 mm。
`ArticulationAction` 组织目标，`apply_action` 下发给驱动器，继续物理步后才有运动。
只控制第一个手指，第二个由 mimic 约束跟随。

分步测试结果：180 步后，张开两指约为 29.29 mm，闭合两指约为 0.38 mm，均在本课 1 mm 容差内。
Script Editor 同款异步入口已通过 MCP 执行，任务正常结束并自动暂停。
这是空载开合验证，不代表已能夹住物体。实际值不必与目标严格相等。

你的实验：将 `OPEN_POSITION_M = 0.03` 改为 `0.015`，重新运行，观察张开幅度是否变小。
下节再让末端到达方块上方，接入 IK / 运动生成。

## 第四课：末端到方块上方

已实现 `scripts/04_hover_above_cube.py`。使用成熟Lula求解器和RMPflow，不手写IK算法。
先暂停前一课，然后执行：

```python
import runpy
runpy.run_path(r"D:\qtproject\IssacSim\scripts\04_hover_above_cube.py", run_name="__main__")
```

自动验证IK、换算抓取中心偏移、运动并在到达后暂停。控制台输出目标/实际TCP和误差。
默认方块中心上方15cm；实验改 `HOVER_HEIGHT_M = 0.20`，观察末端上移约5cm。
15cm测试位置误差约2.75mm，20cm测试约3.05mm；均在5mm阈值内。只有悬停，尚未抓取。
细节、API和验证边界见[学习拼图.md](学习拼图.md)第四块。下一步是下降、夹紧、抬起和抓取判断。

## 第五课：抓住并提起方块

```python
import runpy
runpy.run_path(r"D:\qtproject\IssacSim\scripts\05_pick_cube.py", run_name="__main__")
```

先暂停现有任务。脚本自动Stop恢复场景起点，配置当前会话物理参数，然后完成：到上方 → 下降 → 合拢 → 抬起 → 保持1秒并暂停。
目标抬高12cm，分步测试方块实际抬高约11.91cm；通过接触/摩擦抓取，不绑定或瞬移方块。
本次固定100g方块、摩擦0.8/0.6；连杆惯量按碰撞几何估算，用于解决当前资产接触抖动，不是真机标定参数。
物理配置仅在当前会话生效；最新结果覆盖写入 `outputs/lesson05_latest.json`。

本课只先看 `PickLesson.tick()` 中的五个阶段，不要求一次读懂所有辅助代码。
实验：把 `LIFT_HEIGHT_M = 0.12` 改成 `0.16`，观察方块实际抬升高度。搬运和松开放置见第六课。

## 第六课：完整抓取与放置

先暂停旧任务，再在 Script Editor 执行：

```python
import runpy
runpy.run_path(r"D:\qtproject\IssacSim\scripts\06_pick_and_place.py", run_name="__main__")
```

自动Stop恢复起点，复用第五课抓取后继续：搬运 → 下降 → 松开 → 撤离 → 检查。结束自动暂停，不保存USD。
默认A约为 `(0.18, 0, 0.775)` m，B为 `(0.18, 0.20, 0.775)` m；B的高度由桌面和方块尺寸计算。
放置检查包含实际位置误差、桌面高度、方块速度、夹爪实际张开及撤离距离，连续稳定1秒才成功。

分步及自动运行实测约27.25秒完成，平面放置误差约2.84mm，夹爪撤离约11.89cm；自动结束后保持暂停。
故意屏蔽松爪命令时，程序在Release阶段超时；结果记录在 `outputs/lesson06_negative_no_release.json`。
正常运行最新报告：`outputs/lesson06_latest.json`，每次覆盖写入。

你的小实验：将 `PLACE_XY_M = (0.18, 0.20)` 改为 `(0.18, 0.15)`，观察B点变化并检查报告。
先理解 `tick()` 新增的五个阶段、`set_tcp()`/`grip()` 下发动作、`placement_metrics()` 读取实际结果。
当前只针对桌面5cm方块；没有全连杆碰撞评测或通用抓取规划，IK终点可达不保证整条路径无碰撞。

## 第七课：连续五轮与结果汇总

先暂停旧任务，在 Script Editor 运行：

```python
import runpy
runpy.run_path(r"D:\qtproject\IssacSim\scripts\07_repeat_pick_and_place.py", run_name="__main__")
```

自动重复五轮第六课，每轮Stop恢复起点、重新建立物理句柄、核对方块位置和关节状态，再执行抓取放置。最后一轮结束后保持暂停。
运行中点Pause/Stop会取消整批，保留已产生的报告；不要同时运行其他控制脚本。

报告位置：`outputs/batches/日期_时间_唯一编号/`，每次启动新建文件夹，不覆盖旧实验。

```text
run_001.json    第一轮实际结果、初始状态及重置检查
run_002.json    第二轮
...
run_005.json    第五轮
summary.json   成功率、成功轮次的耗时/误差统计、失败阶段和报告索引
```

`Completed`表示计划轮数全部完成，不等于全部成功。成功率分母为Succeeded/Failed/TimedOut轮数，Cancelled不参与分母；没有可评价轮次时成功率为null。
普通抓取失败/超时会记录并继续下一轮；初始化失败、起点恢复异常或切换场景会结束整批。
当前固定第六课配置重复，不进行随机化或参数扫描；五次同条件成功也不代表任意场景下都能成功。

你的小实验：将第七课顶部 `RUN_COUNT = 5` 改为 `3`，确认新目录中只有三份run报告，并核对summary里的轮数。
只需先理解 `run_one()` 的生命周期和 `batch()` 的外层循环，运动学沿用前几课。

2026-11-12实测：批次 `20261112_170517_006f15f1` 五轮均Succeeded；平均仿真耗时27.25秒，平均平面放置误差2.844mm，各轮起点检查通过，批次结束后暂停。五份独立报告与summary计数已核对。
另测中途Pause：只产生一份Cancelled报告，未启动下一轮；批次运行中重复启动也被拒绝。

## 第八课第一步：导入瓶状刚体（2026-11-13）

已另存并打开 `Isaac_Desk_Arm_Bottle.usda`，原方块场景仍用于第1～7课。
官方YCB芥末瓶位于 `/World/PracticeBottle`，物理刚体在 `/World/PracticeBottle/Body`。
瓶子置于桌面 `(0.18, -0.20)` 附近，原方块保留。原始瓶盖朝-Y，外层绕X旋转-90°后竖直。
尺寸约9.60×5.82×19.13cm，质量约603g，保留资产自带凸分解碰撞；这不是液体或软瓶仿真。
分两次各180物理步检查：瓶子没有倾倒，第二段中心位移约0.013mm；碰撞代理使渲染底部约高于桌面1.7mm，仍有轻微接触速度波动，抓取前需进一步调试。

导入代码和注释：`scripts/08_import_bottle.py`，需在该独立场景Stop状态下运行；已存在则只检查，不重复创建。
资产以官方HTTPS USD引用加载，需要可访问资源及材质依赖。第八课负责导入与落稳；抓取控制见第九课。

后续修复：内部Mesh曾出现约40cm局部平移，导致几何体与重心分离、倾斜悬空。已恢复官方资产内部变换，保持603g质量与原碰撞配置；直立及侧倒落稳测试通过，场景恢复直立并保存。
修复脚本 `scripts/08_repair_bottle_transform.py`；记录 `outputs/bottle_transform_repair.json`。摆放整瓶时选PracticeBottle或物理Body，不要单独移动内部网格。

## 当前启动版本（2026-11-13）

已切换到 `D:/IsaacSim` 的6.1.0-rc.26，使用PhysX。MCP插件复制至 `D:/IsaacSim/IssacSim Mcp/isaacsim-mcp-server`，当前MCP连接到此版本。
打开场景仍为 `Isaac_Desk_Arm_Bottle.usda`；已保存会话显示设置与视角，机械臂、桌面、方块和瓶子加载通过，并完成120物理步检查。
旧教学脚本的全部动作流程尚未在6.1重新验证，不将5.1的测试结论直接视为6.1兼容性结论。启动时ROS2 Bridge失败，当前抓取场景不依赖ROS2，未在此次处理其依赖。
切换前备份在 `outputs/version_switch_backup/`；版本与加载检查记录 `outputs/isaac61_startup/switch_result.json`。

## 第九课：瓶身中部抓取与放置

打开 `Isaac_Desk_Arm_Bottle.usda`，暂停现有任务，在Script Editor执行：

```python
import runpy
runpy.run_path("D:/qtproject/IssacSim/scripts/09_pick_and_place_bottle.py", run_name="__main__")
```

从侧面夹持瓶身中部（TCP距桌面10cm）、抬升12cm、搬运到 `(0.18, 0.20)` m、缓慢松爪、水平退出后上抬、检查放稳。结束自动暂停，结果在 `outputs/lesson09_latest.json`。

使用自定义加长夹指：原指形沿长度放大1.8倍，外观与碰撞几何同步改变，TCP前移36mm。这是仿真夹具配置，不是原版Franka硬件能力。瓶子保留官方603g质量。

当前参数是假设，非实物标定：静/动摩擦1.5/1.5、夹爪驱动上限60N、刚度5000、阻尼100、抗扭接触斑半径10mm、PhysX 240Hz。驱动上限不等于实测指尖力，惯量按碰撞几何估算。瓶子与夹指采用开启Shrink Wrap的凸分解。

轨迹执行采用Lula连续IK参考与限速关节驱动，没有全局避碰保证；原方块课程保持RMPflow。设置仅在当前会话覆盖，不保存USD或修改官方资产。切回方块课程前重新打开原场景。

详解见[学习拼图.md](学习拼图.md)第13节。旧瓶颈基线保存在 `outputs/lesson09_neck_grasp_before_midbody.py` 和 `outputs/lesson09_neck_baseline_success.json`，其36.74秒/3.23mm结果不代表当前中部方案。

2026-11-13中部方案完整单轮验证通过：46.53秒仿真时间，实际最大抬升11.74cm，最终平面放置误差5.31mm、倾角0.50°，完成松爪、撤离并连续放稳1秒后自动暂停。报告 `outputs/lesson09_midbody_success.json`，抓取截图 `outputs/lesson09_midbody_grasp.png`。此前一轮在撤离阶段触发现实时间240秒限制，现等待上限改为600秒；仿真总时限100秒和各阶段22秒限制保留。本次仅验证固定场景单轮，不代表批量成功率。
