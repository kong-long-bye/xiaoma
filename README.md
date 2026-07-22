# xiaoma：UJIIndoorLoc Transformer + SVR

这是对原项目的精简重构版。代码中的关键步骤已经补充中文注释，
包括数据预处理、Transformer 输入形状、训练循环、早停、SVR、
模型保存、Git 分支目录和预测加载流程。

核心流程：

1. 清洗和缩放 WiFi WAP 指纹；
2. 使用 PyTorch 官方 `nn.TransformerEncoder` 编码器提取特征；
3. 使用 scikit-learn 官方 `TransformedTargetRegressor`、
   `MultiOutputRegressor` 和 `SVR` 回归经纬度；
4. 保存模型、指标、预测结果和 UTF-8 日志。

原项目中的手写多头注意力、手写 Transformer Block、Optuna 多阶段搜索、
CatBoost 残差校准、重复存储封装和大量非必要绘图代码已移除。

## 目录

```text
xiaoma/
├── data/
│   ├── trainingData.csv
│   └── validationData.csv
├── src/
│   ├── config.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── model/
│   └── log/
├── environment.yml
├── requirements.txt
└── README.md
```

训练后目录会自动变为：

```text
model/
├── <git分支名>/
│   └── <YYYYMMDD_HHMMSS>/
│       ├── transformer.pt
│       ├── pipeline.joblib
│       ├── config.json
│       ├── metrics.json
│       ├── training_history.csv
│       ├── predictions_validation.csv
│       └── predictions_evaluation.csv
└── log/
    └── <git分支名>/
        └── <YYYYMMDD_HHMMSS>_<git分支名>.log
```

没有 Git 仓库时，分支名使用 `no-git`。
也可以通过 `--branch` 手动指定分支名称。

## 推荐：使用 Conda 创建环境

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate xiaoma
```

确认环境：

```bash
python --version
python -c "import torch, pandas, sklearn; print(torch.__version__)"
```

以后修改了 `environment.yml`，可以更新现有环境：

```bash
conda env update -f environment.yml --prune
```

不再需要该环境时：

```bash
conda env remove -n xiaoma
```

### NVIDIA GPU

`environment.yml` 默认不固定具体 CUDA 版本，因为不同机器的显卡驱动和
CUDA 兼容情况不同。需要 GPU 时，应在创建环境后，根据本机环境安装对应的
PyTorch CUDA 构建；代码中的 `--device auto` 会在 CUDA 可用时自动使用 GPU。

当前项目有两种训练模式：`patch` 是 CPU 友好的快速/轻量模式；`wap` 是论文复现模式，
每个 WAP 特征作为独立 token，训练时需要 CUDA/GPU。

## requirements.txt 是否还能使用

可以。`requirements.txt` 并不与 Conda 冲突，可以先创建 Conda 环境，
再在环境中使用 pip 安装：

```bash
conda create -n xiaoma python=3.11
conda activate xiaoma
python -m pip install -r requirements.txt
```

不过本项目更推荐 `environment.yml`，因为它同时记录：

- Conda 环境名称；
- Python 版本；
- Conda channels；
- Python 依赖。

`requirements.txt` 继续保留，适合没有 Conda 的服务器、Docker 或 CI 环境。

## 快速运行检查

该命令只读取少量数据，用于确认数据加载、训练、SVR、模型保存和日志流程正常：

```bash
python -m src.train \
  --smoke-test \
  --epochs 1 \
  --batch-size 32 \
  --device cpu \
  --token-mode patch \
  --patch-size 16 \
  --branch smoke-test
```

Windows PowerShell 可以写成一行：

```powershell
python -m src.train --smoke-test --epochs 1 --batch-size 32 --device cpu --token-mode patch --patch-size 16 --branch smoke-test
```

## 完整训练

CPU / patch 模式：

```bash
python -m src.train \
  --token-mode patch \
  --patch-size 16 \
  --device cpu \
  --epochs 30 \
  --batch-size 128 \
  --building-id -1
```

GPU / wap 论文复现模式：

```bash
python -m src.train \
  --token-mode wap \
  --device auto \
  --epochs 30 \
  --batch-size 128 \
  --building-id -1 \
  --svr-c 100 \
  --svr-epsilon 0.1
```

`wap` 模式会在没有 CUDA/GPU 时直接报错；没有 GPU 时请使用 `patch` 模式。

## 预测

使用当前 Git 分支下最新模型：

```bash
python -m src.predict \
  --input data/validationData.csv \
  --model-dir latest
```

指定模型目录：

```bash
python -m src.predict \
  --input data/validationData.csv \
  --model-dir model/main/20260714_120000 \
  --output model/main/20260714_120000/my_predictions.csv
```

预测 CSV、JSON 和日志文件均使用 UTF-8 编码。

## CPU 线程设置

项目默认将 PyTorch CPU 线程数设置为 1，避免部分机器因为线程过多而变慢。
需要增加线程时，可以设置环境变量：

Linux / macOS：

```bash
export XIAOMA_TORCH_THREADS=4
python -m src.train
```

Windows PowerShell：

```powershell
$env:XIAOMA_TORCH_THREADS=4
python -m src.train
```


# Windows 版 README（建议替换原 README）

## 环境

```powershell
conda env create -f environment.yml
```

```powershell
conda activate xiaoma
```

## 快速测试（一行）

```powershell
python -m src.train --smoke-test --epochs 1 --batch-size 32 --device cpu --token-mode patch --patch-size 16 --branch smoke-test
```

## 正式训练（一行）

CPU / patch 模式：

```powershell
python -m src.train --token-mode patch --patch-size 16 --device cpu --epochs 30 --batch-size 128
```

GPU / wap 论文复现模式：

```powershell
python -m src.train --token-mode wap --device auto --epochs 30 --batch-size 128 --branch wap-gpu
```

指定建筑训练：

```powershell
python -m src.train --token-mode patch --device cpu --building-id 0
```

## 预测（一行）

```powershell
python -m src.predict --input data\validationData.csv --model-dir latest
```

```powershell
python -m src.predict --input data\validationData.csv --model-dir latest --output result.csv
```

## 输出

模型：

```text
model/<branch>/<time>/
```

日志：

```text
model/log/<branch>/<time>_<branch>.log
```

参数	说明
```
--train-csv	训练集 CSV 路径
--eval-csv	官方评估集 CSV 路径
--model-root	模型产物、预测和日志的输出根目录
--building-id	建筑编号筛选（-1 = 全部，0/1/2 = 指定建筑）
--branch	运行分支名，None 则自动生成时间戳
--device	PyTorch 设备（auto 优先 CUDA）
--token-mode	训练模式：patch 可在 CPU 上运行；wap 是论文复现模式，需要 GPU
--patch-size	patch 模式下每个 token 包含的连续 WAP 特征数量
--epochs	自编码器最大训练轮数
--batch-size	训练 mini-batch 大小
--feature-batch-size	提取 latent 特征的推理 batch 大小
--learning-rate	AdamW 初始学习率
--svr-c	SVR 正则化参数 C
--svr-epsilon	SVR 的 epsilon 管道宽度
--svr-max-train-samples	SVR 最大训练样本数（0 = 全量）
--smoke-test	开启冒烟测试快速验证全流程
--smoke-train-rows	冒烟测试训练集行数
--smoke-eval-rows	冒烟测试评估集行数
```
