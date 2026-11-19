# Dataset Studio

Isaac Sim 内的双视图数据采集面板，使用 `omni.ui`，提供独立相机多角度、瓶子位置和灯光随机化参数控制。

打开项目场景后，在 Isaac Sim 的 Script Editor 中运行：

```python
import runpy

runpy.run_path(
    r"D:\qtproject\IssacSim\tool\dataset_studio\panel.py",
    run_name="__main__",
)
```

`panel.py` 是面板入口，需要在 Isaac Sim 中运行。项目根目录通过脚本位置自动解析。

面板复用项目的数据采集功能：

- `../../src/bottle_sim/dataset.py`：执行采集和质量检查。
- `../../src/bottle_sim/randomization.py`：随机采样与图库生成。
- `../../configs/dataset_randomization.json`：参数配置。
- `../../outputs/bottle_dataset/`：采集输出。

详细参数和录屏说明见 [使用文档](../../docs/dataset_studio.md)。
