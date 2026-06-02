# 开发指引

## 项目概览

本仓库是一个基于 PyTorch 的训练框架，支持模块化的模型定义、多个数据集类型、可配置的训练流程，以及常见的训练/评估/导出工具。

## 项目结构

```
ignet_ai_train/
├── __main__.py           # 模块入口，支持 python -m ignet_ai_train
├── cli.py               # 命令行接口解析
├── config.py            # YAML 配置管理与加载
├── model_builder.py     # 模型构建逻辑
├── dataset/             # 数据集相关模块
│   ├── create.py        # 数据集工厂函数
│   └── modules/         # 数据集类型实现（图像、音频、NPY）
├── nn/                  # 神经网络模块
│   ├── create.py        # 网络创建工厂
│   └── modules/         # 自定义网络层和架构实现
├── optim/               # 优化器相关模块
│   └── modules/         # 自定义优化器实现（如 Muon）
└── utils/               # 通用工具函数
    ├── train.py         # 训练流程工具
    ├── eval.py          # 评估流程工具
    └── ...

test/                      # 示例脚本与示例配置
├── build_model.py        # 模型构建测试脚本
├── evaluate_classify.py  # 分类评估示例
├── export.py             # 模型导出示例
├── test.py               # 综合测试脚本
├── train.py              # 训练示例脚本
├── train.yaml            # 训练配置示例
├── train_net.yaml        # 训练网络示例配置
└── models/               # 示例模型 YAML 配置
```

## 依赖同步

项目使用 `pyproject.toml` 定义依赖，`uv.lock` 作为锁文件，推荐使用 `uv` 进行依赖管理。

- 安装依赖：
  ```shell
  uv sync
  uv pip install -e .
  ```
- 添加新依赖：
  - `uv add <package>`
- 升级依赖：
  - `uv sync --upgrade`

如果仓库中存在 `.venv`，建议先激活虚拟环境：

- PowerShell: `.venv\Scripts\Activate.ps1`
- CMD: `.venv\Scripts\activate.bat`
- Linux: `source .venv/bin/activate`

> 不要直接使用 `pip install` 安装生产依赖，除非你清楚当前环境与 `uv` 锁文件的同步关系。

## 运行方式

### 直接运行包入口

```powershell
python -m ignet_ai_train --config test/train.yaml
```

### 运行示例脚本

仓库中没有配置标准测试框架，常见的开发/验证方式是直接运行 `test/` 目录下的脚本：

```powershell
python test/train.py --config test/train.yaml
python test/evaluate_classify.py
python test/export.py
python test/build_model.py
```

### 注意

1. `main.py` 目前是一个空桩文件，不建议作为入口调用。
2. 避免提交时引入不必要的依赖。
3. 测试用的依赖包建议用 ```uv pip install <package>```, 避免引入测试所需的依赖

## 测试与示例文件位置

本仓库目前没有 pytest 或 unittest 等标准测试框架，测试相关内容集中在 `test/` 目录下：

- `test/train.py` - 训练示例
- `test/evaluate_classify.py` - 分类评估示例
- `test/export.py` - 模型导出示例
- `test/build_model.py` - 模型构建验证
- `test/test.py` - 综合测试脚本
- `test/models/` - 多个模型 YAML 配置示例

如果后续引入正式测试框架，建议：

- 新增 `tests/` 或 `test_cases/` 目录
- 使用 `pytest` 或 `unittest` 编写可重复执行的单元测试
- 保留 `test/` 目录作为示例脚本和玩法演示

## 开发常见流程

1. 激活虚拟环境
2. 同步依赖：`uv sync`
3. 修改代码或新增模块
4. 运行示例脚本验证功能
5. 如需新增依赖，使用 `uv add <package>`
6. 


## 重要目录说明

- `ignet_ai_train/` - 核心包代码
- `ignet_ai_train/dataset/` - 数据集加载与工厂逻辑
- `ignet_ai_train/nn/` - 自定义网络层与模型构建
- `ignet_ai_train/optim/` - 自定义优化器实现
- `ignet_ai_train/utils/` - 训练、评估、导出等工具
- `test/models/` - YAML 模型配置示例

## 环境变量说明

本项目当前对环境变量的主要支持是 `IGNET_HOME`：

- `IGNET_HOME` 用于指定 IGNet 的主目录，默认值是 `.ignet_`。
- 该目录下会创建 `cache/` 子目录，用于存放下载缓存和资源文件。
- 在 Windows PowerShell 中设置示例：
  - `$env:IGNET_HOME = 'D:\\work\\ignet_cache'`
- 在 Linux/macOS 或 Git Bash 中设置示例：
  - `export IGNET_HOME=$HOME/.ignet_`

如果你需要让配置加载或缓存路径在不同环境间保持一致，可通过此环境变量覆盖默认位置。

## 其他注意事项

- 配置驱动：模型结构、训练参数和优化器通常由 YAML 文件定义
- URL 支持：模型 YAML 和数据集路径可以是本地文件路径，也可以是 HTTP URL
- 代码注释：很多功能实现分散在模块内部，必要时请阅读对应源码注释
- CI/静态分析：当前仓库没有配置自动化 CI、格式化或静态类型检查工具，开发时可先按现有风格维护
