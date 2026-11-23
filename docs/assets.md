# 场景与资产准备

仓库中的 `Isaac_Desk_Arm_Bottle.usda` 保存场景布局、相机、灯光和 ROS2 图。Franka 与隐藏回退芥末瓶使用 NVIDIA 官方 HTTPS USD 引用，需要能访问这些资源。

当前主对象是 Lightwheel OpenSource 的 `bottle014`。第三方原始网格和纹理不随本仓库发布；请从原提供方取得资产并按其许可使用。

本地准备步骤：

1. 在 Lightwheel 场景中定位 `/root/bottle014`，单独导出它及瓶身、瓶盖和约束，保持刚体层级。
2. 使用外层 Xform 调整原点到瓶底；保留引用后路径 `/World/PracticeBottle/bottle014/body` 和 `/World/PracticeBottle/bottle014/lid`。
3. 将导出的资产放在 `assets/waterbottle_only_bottom_origin.usda`。检查它引用的纹理全部可以解析，移机时将纹理一起收集，并将内部绝对路径改成相对路径。
4. 打开仓库根目录的 `Isaac_Desk_Arm_Bottle.usda`。核对完整水瓶尺寸约 85.5 × 85.5 × 288.6 mm、桌面 Z=0.75 m、瓶子初始 XY=(0.18,-0.20) m。
5. 核对 `/World/Franka/panda_hand/WristCamera` 能完整看到瓶身和瓶盖，再运行视觉脚本。pick_task.py 在抓取初始化时设置动态瓶身和 560 g 质量基线。

项目并非只靠 Python 文件即可独立运行：主水瓶 USD、材质纹理与 Isaac 官方机器人资产均需正确解析。保留场景的初始姿态，不要把正在抓取时的关节/物体姿态另存成新起点。
