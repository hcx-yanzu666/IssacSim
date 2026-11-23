# Outputs 目录说明

`outputs/` 的清理目标是只保留模型训练数据和训练产物：`bottle_dataset/` 与 `yolo_train/`。运行任务后，脚本会重新创建对应的实验输出目录。

| 路径 | 用途 | 保留原因 |
| --- | --- | --- |
| `bottle_dataset/` | 采集的 RGB、Mask、深度和训练标签 | 训练与数据集面板的默认输出/输入 |
| `yolo_train/` | YOLO 权重和训练记录 | 推理脚本与抓取默认模型依赖 |
| `yolo_pick/` | 每轮视觉抓取结果、图像和 JSON | 运行时自动创建；成功轮次只保留最小证据集 |
| `evaluation_report.html` | 抓取结果汇总页 | 每轮任务结束自动刷新，也可手动重建 |
| 其他目录和散落文件 | 相机样例、课程记录、诊断与备份 | 清理后不保留；可通过重新运行实验生成新结果 |

生成评测报告：

```powershell
python tools\build_evaluation_report.py
```

数据集采集继续写入 `bottle_dataset/`，训练继续写入 `yolo_train/`。抓取时会在 `yolo_pick/<run_id>/` 创建单轮结果；成功轮次只保留 `result.json`、RGB、YOLO Mask 和位姿估计 JSON。失败轮次保留中间文件，供排查使用。
