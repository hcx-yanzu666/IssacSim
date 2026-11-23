# 任务评测报告

每轮视觉抓取结束后会自动刷新 HTML。需要单独重新生成时，在项目根目录运行：

```powershell
python tools\build_evaluation_report.py
```

默认扫描 `outputs/yolo_pick/*/result.json`，生成 `outputs/evaluation_report.html`，双击即可打开。网页离线可用，不依赖 CDN 或 Web 服务。

页面分三部分：

- 概况：轮数、完整任务成功率、抓取验证通过率、放置误差中位数。
- 图表：累计完整成功率、单轮放置误差、最大搬运滑移折线图，以及失败阶段条形图。悬停数据点可查看运行编号和数值。
- 单轮记录：按成功/失败筛选，展开查看失败原因、RGB、Mask、预测图和任务配置。

成功运行现在只保留 `result.json`、`rgb.png`、`yolo_mask.png` 和 `bottle_pose_estimate.json`。相机参数、Mask 统计、模型哈希和推理信息写入 `result.json`。深度数组、worker 日志、请求/响应 JSON、预测叠加图和定位预览仅在任务失败时保留；成功后自动清理。生成器兼容读取旧轮次中独立的 `yolo_result.json`。

JSON 是原始数据，HTML 是生成时的快照；新增运行后重新执行生成命令即可更新。图表和汇总数据内嵌在 HTML 中，图片与 JSON 使用相对链接，移动或分享报告时需一起保留对应 `yolo_pick/` 目录。

历史调试轮次可通过 `configs/evaluation_exclusions.json` 或 `--exclude-run-id` 排除，报告页显示排除数量。真实任务失败应保留，不要为了提高成功率移除失败记录或修改原始状态。对比参数时使用相同权重和随机种子，并一次只改一个实验变量。

也可指定输入目录和输出文件：

```powershell
python tools\build_evaluation_report.py --run-root outputs\yolo_pick --output outputs\evaluation_report.html
```

## 统计口径

完整成功率仅将 `Succeeded` 计为成功；抓取通过率只使用明确记录 `pick_verified` 的轮次，卡片显示有效记录数量。缺失指标显示为 `—`，图表断开，不按零计算。定位误差与放置误差分别读取定位报告和任务结果，不相互代替。图中的参考线为当前配置：放置 XY 误差 20 mm、搬运滑移 25 mm。它们不用于重新判定历史结果；历史运行仍保留当时状态。改变判定标准后的运行不能当作同条件重复实验，修改控制器阈值后也需同步图表参考线。

当前报告源数据包含历史调试记录，轮数代表已保存的运行记录，不代表同条件重复实验。定位误差来自仿真真值对照字段，仅用于仿真内评估。模型指纹保留在每轮原始 JSON 中，不在 HTML 汇总页展示。
