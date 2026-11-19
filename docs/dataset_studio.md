# 域随机化采集与录屏面板

## 打开面板

在 Isaac Script Editor 中执行：

```python
import runpy
runpy.run_path(
    r"D:\qtproject\IssacSim\tool\dataset_studio\panel.py",
    run_name="__main__",
)
```

面板左边是独立采集相机 `/World/DatasetCaptureCamera`，每张图片随机取景；右边是固定斜侧场景视角。它使用 Isaac 原生 `omni.ui` 和 `ViewportWidget`，无需安装 Qt。视图相机只创建在 session layer，不修改原资产或手眼相机的安装参数。

## 可以调整什么

| 控件 | 默认范围 / 值 | 意义 |
|---|---|---|
| Bottle X offset | -25～25 mm | 相对原始 world X=0.18 m 的位置偏移 |
| Bottle Y offset | -15～15 mm | 相对原始 world Y=-0.20 m 的位置偏移 |
| Camera azimuth | -90～-30° | 桌面前侧观察区域，减少机械臂遮挡；0°为世界+X，90°为+Y |
| Camera elevation | 20～50° | 相机在目标水平面上方的仰角，对应向下观察瓶子 |
| Camera distance | 0.55～0.85 m | 相机到瓶身观察中心的距离 |
| Light azimuth | -140～-40° | 灯相对于固定桌面目标的水平角，0°为世界+X方向 |
| Light elevation | 45～75° | 灯在水平面之上的仰角 |
| Desk light intensity | 1000～2400 | USD 灯光强度，不等于实测 lux |
| Ambient intensity | 400～850 | 环境光强度 |
| Light distance | 1.6 m | 球形光源到固定桌面目标的距离 |
| Hold each sample | 1.5 s | 合格图片保存后保持画面，便于录屏；0 表示快速批量采集 |
| Samples | 50 | 正式批量采集目标数量，可先改成 12 |
| Master seed | 20261120 | 控制每轮参数抽样序列 |

当前 DeskLight 是 SphereLight，改变旋转不会产生期望的入射方向变化。因此把方位角、仰角和距离转换成灯的位置：`target + r × [cos(el)cos(az), cos(el)sin(az), sin(el)]`。固定 target 为瓶子基准位置上方 14 cm，瓶子偏移与灯位置独立采样。灯的实际世界位置会写入每张图的元数据。

独立相机也按球坐标采样位置，但观察目标跟随本张图片的瓶子 XY，目标高度为桌面上方14.5 cm。使用 `Gf.Matrix4d.SetLookAt(...).GetInverse()` 得到相机世界变换，USD 相机沿局部 -Z 看向目标。焦距固定18 mm，水平/垂直光圈为24/13.5 mm，分辨率640×360。相机姿态在应用参数后、渲染之前更新，RGB、深度、Mask 均使用同一渲染产品。

YOLO 训练只需要 RGB 与对应标签。相机内外参作为轻量元数据保存，不是 YOLO 的训练输入；固定光学参数下内参不变，外参随每次相机位姿变化，用于后续深度反投影、检查和复现。独立相机的任意采样视角不代表机械臂手眼相机实际可达，也尚未加入相机位置的机器人碰撞检查。

场景自带的 `/Environment/defaultLight` 是强方向光，会掩盖随机灯光变化，采集期间临时隐藏它，结束后恢复。可在 JSON 的 `mute_environment_light` 中控制。

配置保存在 `configs/dataset_randomization.json`。修改面板数值只影响下一次启动；点击 **Save settings** 才写入配置文件。采集期间不要手动移动场景或播放仿真。

## 按钮与录屏

1. **Preview 1**：随机采一个合格样本，保持至少 3 秒，随后恢复场景。预览数据单独存放，不覆盖既有批次。
2. **Start capture**：采集完整批次。先 Stop 复位机器人和瓶子，固定瓶子姿态，随后只改位置与光照，不执行抓取。
3. **Stop**：请求停止；等待当前渲染返回，保留已保存图片，生成部分结果报告，并恢复场景参数。
4. **Open gallery**：打开当前批次的本地 HTML 页面。可浏览缩略图、原图/分割轮廓对照，点击 **Play 1.5s** 自动播放，适合录第二段视频。

建议 Samples=12、Hold=1.5，先录左右视图变化，再打开 gallery 录原图和标注切换。每个样本除保持时间外还有渲染耗时，因此完整采集时长会大于样本数×1.5 秒。录屏过程中可折叠 Capture settings，保留视图和参数摘要。

停止或结束后，场景恢复原灯光、瓶子位置和独立相机的采集前位姿，状态栏会明确提示。不要把恢复后的当前画面当成最后一张训练图片；保存结果以 gallery 为准。

## 数据和质量检查

每批生成：

- `images/`、`labels/`、`masks/`、`depth/`：RGB、YOLO 分割多边形、实例 Mask、米制轴向深度。
- `metadata/`、`samples.jsonl`：种子、所有抽样值、灯位置、瓶身/瓶盖像素数量、相机内外参。
- `config.json`、`summary.json`：实际使用的完整配置、合格数量和拒收原因。
- `previews/`、`contact_sheet.jpg`、`gallery.html`：用于检查、展示和录屏的预览，不作为训练输入。

自动检查瓶身与瓶盖是否可见、是否出画面、Mask/框是否太小、有效深度比例是否至少95%、多边形是否有效。当前最大尝试次数为目标数量×5，耗尽后标为 incomplete，不把不足数量写成完成。

默认位置范围比旧版更宽，出画面样本会被拒收。因此保存数据的实际位置分布是经过筛选的分布，不是原始均匀分布。多角度模式下相机朝向跟随瓶子，更容易完整取景；过近距离、极端角度或遮挡仍可能被拒收。本次没有改变现有手眼相机安装参数。

同一种子复现的是参数序列；不保证不同 GPU/渲染版本逐像素一致。每个参数使用独立子随机流。当前 train/val 按采集顺序划分，同属一个场景分布；仍需另采独立条件的测试集，不能把本批验证分数当作真实环境泛化效果。

## 无面板调用

仍可直接运行第14步，默认从 JSON 读取配置：

```python
import runpy
module = runpy.run_path(r"D:\qtproject\IssacSim\src\bottle_sim\dataset.py")
task = module["start"](config={"target_samples": 50, "demo_hold_seconds": 0})
```

## 固定相机版本验证（历史）

最终灯光方案批次：`outputs/bottle_dataset/batch_20261120_20261119_140825_77d2a1`。

- 18 次尝试，12 张合格（9 train、3 val），6 次因画面边缘截断拒收。
- 12 张 RGB、Mask、深度、标签尺寸和索引一致，瓶身与瓶盖均有覆盖。
- 标签多边形栅格化后与原始 Mask 的最小 IoU 约 99.10%。
- 相同种子再次采集时抽样参数一致；灯光几何距离检查、非法范围拒绝检查通过。
- 主动停止测试保存了1张后取消，瓶子位置、灯位置/强度、固定方向光可见性及两个刚体的运动学属性均恢复到测试前值。

第一版未加入背景、材质、遮挡、深度噪声或物理参数随机化，也未修改 YOLO 模型和抓放状态机。
