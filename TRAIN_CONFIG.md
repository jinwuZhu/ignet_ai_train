## 训练文件配置说明

本文档说明工程中用于训练的配置文件格式、常用字段、示例模板以及调试/运行建议，目标是帮助开发者快速理解并自定义训练流程。

---

**位置说明**

- 默认/示例配置文件：[test/train.yaml](test/train.yaml)
- 网络训练示例：[test/train_net.yaml](test/train_net.yaml)
- 模型定义示例：位于 [models/](models/) 目录（例如 [models/MobileNet.yaml](models/MobileNet.yaml)）
- CLI 使用入口：[ignet_ai_train/cli.py](ignet_ai_train/cli.py)

---

**配置文件格式**

- 本项目使用 YAML 格式的配置文件（`.yaml` 或 `.yml`），请保持缩进一致（建议使用两个空格）。
- 配置文件以顶层字典形式组织，通常包含 `dataset`、`model`、`optimizer`、`training`、`logging`、`device` 等块。

示例顶层结构：

- `dataset`: 数据集路径、预处理、类别数等
- `model`: 模型类型及其超参（可引用 `models/*.yaml` 中的定义）
- `optimizer`: 优化器类型与学习率等
- `training`: 训练相关的超参（epoch、batch、lr_scheduler、checkpoint 等）
- `augmentation`: 数据增强相关配置（可选）
- `resume`: 从 checkpoint 恢复训练的路径（可选）
- `seed`: 随机种子（可选）
- `logging`: 日志和可视化（TensorBoard、保存路径）

---

**字段详解（常见字段）**

- `dataset`:
	- `type`: 数据集类型（如 `ImageFolder`, `NpyFolder`, `AudioFolder` 等，参见 `ignet_ai_train/dataset/modules/`）
	- `path`: 数据根目录或文件列表
	- `batch_size`: 每个批次大小（训练/验证可分开配置）
	- `num_workers`: 数据加载进程数
	- `input_size`: 输入图像/样本尺寸（例如 `[3,224,224]` 或 `224`）
	- `shuffle`: 是否打乱训练集

- `model`:
	- `name`: 模型名或类型（如 `MobileNetV1`, `ResShift` 等），可与 `models/*.yaml` 对应
	- `params`: 模型特定参数字典（通道数、层数、宽度系数等）
	- `pretrained`: 是否加载预训练权重（布尔或路径）

- `optimizer`:
	- `type`: 优化器类型（如 `SGD`, `Adam`）
	- `lr`: 初始学习率
	- `momentum`:（若适用）动量参数
	- `weight_decay`: 权重衰减系数

- `lr_scheduler`（通常在 `training` 中）:
	- `type`: 调度器类型（如 `StepLR`, `CosineAnnealingLR`, `MultiStepLR`）
	- `params`: 调度器参数（如 `step_size`, `gamma` 或 `milestones`）

- `training`:
	- `epochs`: 总训练轮数
	- `save_interval`: 保存 checkpoint 的轮数间隔
	- `validate_interval`: 验证频率（轮或步）
	- `checkpoint_dir`: checkpoint 保存目录
	- `gradient_clip`: 梯度裁剪阈值（可选）
	- `amp`: 是否使用混合精度训练（如 `true`/`false`）

- `augmentation`:
	- 结构依赖于实现（常见项：`random_crop`, `resize`, `flip`, `normalize`）
	- 可指定概率、参数值和顺序

- `logging`:
	- `log_dir`: 日志保存目录
	- `tb`: 是否启用 TensorBoard
	- `print_freq`: 控制控制台打印间隔

- `device`:
	- `cuda`: 使用 GPU（布尔或 GPU id 列表）
	- `num_gpus`: 多 GPU 使用数量（如果支持）

---

**示例：分类训练配置（最小示例）**

```yaml
dataset:
	type: ImageFolder
	path: data/imagenet
	batch_size: 64
	num_workers: 8
	input_size: 224

model:
	name: MobileNetV1
	params:
		width_mult: 1.0
	pretrained: null

optimizer:
	type: SGD
	lr: 0.01
	momentum: 0.9
	weight_decay: 1e-4

training:
	epochs: 90
	save_interval: 5
	validate_interval: 1
	checkpoint_dir: checkpoints/
	amp: false

lr_scheduler:
	type: StepLR
	params:
		step_size: 30
		gamma: 0.1

logging:
	log_dir: logs/run1
	tb: true
	print_freq: 50
```

**示例：从远程配置加载**

如果使用远程/网络配置（如 `test/train_net.yaml`），通常会先下载或通过 URL 加载，再传入训练脚本。示例 CLI 调用见下文。

---

**常见用法 / CLI**

- 通过项目顶层入口运行训练脚本（示例）：

```bash
python -m igent_ai_train --config test/train.yaml
python -m igent_ai_train --config test/train_net.yaml
```

- `ignet_ai_train/cli.py` 包含参数解析逻辑，支持傳入 `--config`（配置文件路径）和覆盖某些字段的命令行参数（请查看该文件以了解可用选项）。

---

**调试与最佳实践**

- 开发阶段先用较小的 `batch_size` 和 `epochs` 快速跑通配置。
- 如果出现 OOM（显存不足），尝试减小 `batch_size` 或开启 `amp`（混合精度）。
- 对于可复现性：设置 `seed` 并在配置中固定 `num_workers` 和 `pin_memory` 行为。
- 使用 `resume` 字段从上次 checkpoint 继续训练：

```yaml
resume: checkpoints/run1_epoch_20.pth
```

- 日志与可视化：开启 TensorBoard（`logging.tb: true`）并指定 `log_dir`。

---

**故障排查（常见问题）**

- 配置语法错误：YAML 缩进错误最常见，使用在线或本地 linter 校验。
- 模型与数据尺寸不匹配：检查 `input_size` 与模型期望的输入形状。
- 优化器/调度器参数未生效：确认参数名匹配训练实现中的解析键名。

---
