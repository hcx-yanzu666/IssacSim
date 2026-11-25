# 简易室内抓取场景

打开项目根目录 `Isaac_Desk_Arm_Bottle_Room.usda`。它以原场景为 USD 子层，新增木纹长桌外观、背景柜、地板、两面墙及柔和灯光；原 `Isaac_Desk_Arm_Bottle.usda` 保留作回退，两份文件需位于同一目录。

家具引用本机 `D:/tools/Lightwheel_OpenSource/Lightwheel_OpenSource/`，未复制或修改资产库。桌子使用 `Locomotion/Apartment/Props/SM_table_02_0.usd`，背景柜使用 `Locomotion/KitchenRoom/Kitchen_Cabinet001/Kitchen_Cabinet001.usd`。

原 `/World/DeskTop` 仅隐藏外观，碰撞仍启用，尺寸 1.4 × 0.8 m、顶面 0.75 m；新桌视觉尺寸与它重合。新增家具不启用刚体、关节和碰撞。机器人、水瓶、手眼相机及任务参数未改动。家具仅作为环境外观，不能用于新增接触任务。

`/World/Room/OverviewCamera` 为总览相机，原手眼相机保持不变。需要自由浏览时，在视口相机菜单选择 Perspective。

已检查总览、手眼相机画面、桌面高度和任务坐标；未在新背景下重新进行 YOLO 推理或抓放验证。后续数据采集可使用这个场景，已有白桌数据训练的模型仍需在新背景下验证。

`tools/setup_simple_room.py` 保存搭建逻辑，在 Isaac 环境运行 `build()` 可生成此场景；若目标文件已存在会拒绝覆盖。
