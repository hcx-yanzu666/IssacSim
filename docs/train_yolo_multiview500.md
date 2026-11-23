# 多视角瓶子数据集：手动训练 YOLO 分割模型

本批数据路径：

`outputs/bottle_dataset/batch_20261121_20261119_152056_f66af9`

已完成500张：400张训练、100张验证，500次尝试、0次拒收。全量检查确认图片、Mask、深度及标签对应完整，训练/验证集无完全重复图片，标签多边形与原始Mask的最小IoU为98.05%。详情见批次内的 `summary.json`、`validation.json`；抽样总览为 `audit_contact_sheet.jpg`。

当前验证集与训练集来自相同场景及随机化分布，不是独立测试集，也不是现实环境评测。

## 1. 准备

等待采集完成，检查图库。训练前建议关闭 Isaac Sim，释放 GPU 显存。

在 Windows PowerShell 中执行，不要在 Isaac Script Editor 或 Ubuntu 中执行：

```powershell
Set-Location 'D:\qtproject\IssacSim Pj'
& '.\.venv\Scripts\python.exe' -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

本项目虚拟环境已安装 PyTorch 和 Ultralytics。下面直接调用虚拟环境中的程序，无需执行 Activate.ps1。

## 2. 训练

复制下面这一整行：

```powershell
& '.\.venv\Scripts\yolo.exe' segment train model='yolo11n-seg.pt' data='outputs/bottle_dataset/batch_20261121_20261119_152056_f66af9/data.yaml' epochs=80 imgsz=640 batch=4 device=0 workers=0 seed=42 project='outputs/yolo_train' name='multiview500' patience=20
```

参数说明：

| 参数 | 作用 |
|---|---|
| `segment train` | 训练实例分割模型，输出瓶子 Mask |
| `model=yolo11n-seg.pt` | 从项目中已有的预训练小模型开始，不覆盖之前50张训练的权重 |
| `data=.../data.yaml` | 数据集配置，指向图片、标签和 bottle 类别 |
| `epochs=80` | 最多训练80轮，每轮遍历训练集 |
| `imgsz=640` | 训练输入尺寸，原图会按框架规则缩放和补边 |
| `batch=4` | 每批4张，适合先在当前显卡上保守启动 |
| `device=0` | 使用第一张 CUDA 显卡 |
| `workers=0` | 数据加载不启用额外工作进程，简化 Windows 环境运行 |
| `seed=42` | 控制训练随机过程，和采集种子不是同一个概念 |
| `patience=20` | 验证指标连续20轮无改善时提前停止 |
| `project`、`name` | 训练结果保存位置；重复运行时框架通常自动增加目录后缀 |

出现 CUDA out of memory 时，先确认 Isaac 已关闭，再把 `batch=4` 改成 `batch=2`。不需要重新采集数据。

## 3. 看结果

首轮预计输出到 `outputs/yolo_train/multiview500/`，以终端的实际保存路径为准：

- `weights/best.pt`：验证表现最好的模型，后续推理使用它。
- `weights/last.pt`：最后一轮模型，可用于恢复训练。
- `results.png`、`results.csv`：训练损失和验证指标。
- `val_batch*_pred.jpg`：验证图片上的模型预测，检查是否漏检、Mask是否贴合瓶子。
- `args.yaml`：本次训练使用的参数。

同时观察框指标 `(B)` 与掩膜指标 `(M)`，本项目尤其关注 Mask。验证 mAP 高只说明对当前验证分布效果较好，仍需单独检查实际手眼相机视角。

## 4. 手动推理查看 Mask

训练结束后执行（如果训练输出带数字后缀，请相应修改模型路径）：

```powershell
& '.\.venv\Scripts\yolo.exe' segment predict model='outputs/yolo_train/multiview500/weights/best.pt' source='outputs/bottle_dataset/batch_20261121_20261119_152056_f66af9/images/val' imgsz=640 conf=0.25 device=0 retina_masks=True save=True project='outputs/yolo_predict' name='multiview500_val'
```

这一步验证模型可加载和生成预测，不自动替换机器人抓取模型。确认预测后，再将新权重接入现有视觉定位与抓取流程。
