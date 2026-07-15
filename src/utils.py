from __future__ import annotations

"""项目公共工具函数。

这里集中处理随机种子、日志、Git 分支名称、运行目录和评估指标。
训练和预测入口只负责业务流程，不再重复这些基础代码。
"""

import json
import logging
import os
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


LOGGER_NAME = "xiaoma"


@dataclass(frozen=True)
class RunPaths:
    """一次训练或预测任务对应的路径信息。"""

    branch: str
    timestamp: str
    artifact_dir: Path
    log_file: Path


def safe_name(value: str) -> str:
    """将分支名或后缀转换成适合文件夹和文件名的安全字符串。"""

    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    return cleaned.strip("-._") or "unknown"


def get_git_branch(project_root: Path) -> str:
    """获取当前 Git 分支名。

    获取顺序:
    1. CI 环境变量 GIT_BRANCH / BRANCH_NAME；
    2. git branch --show-current；
    3. 无 Git 环境时返回 no-git。
    """

    env_branch = os.environ.get("GIT_BRANCH") or os.environ.get("BRANCH_NAME")
    if env_branch:
        return safe_name(env_branch.removeprefix("origin/"))

    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return safe_name(result.stdout.strip() or "detached")
    except (OSError, subprocess.SubprocessError):
        return "no-git"


def now_timestamp() -> str:
    """返回用于目录和日志文件名的本地时间字符串。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_paths(
    model_root: Path,
    project_root: Path,
    branch: str | None = None,
    log_suffix: str | None = None,
    create_artifact_dir: bool = True,
) -> RunPaths:
    """为一次任务创建模型目录和 UTF-8 日志路径。

    训练产物:
        model/<branch>/<timestamp>/

    日志:
        model/log/<branch>/<timestamp>_<branch>.log
    """

    branch_name = safe_name(branch) if branch else get_git_branch(project_root)
    timestamp = now_timestamp()

    artifact_dir = model_root / branch_name / timestamp
    log_dir = model_root / "log" / branch_name

    # 预测任务通常只需要日志，不需要创建一个空的模型产物目录。
    if create_artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    log_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{safe_name(log_suffix)}" if log_suffix else ""
    log_file = log_dir / f"{timestamp}_{branch_name}{suffix}.log"

    return RunPaths(
        branch=branch_name,
        timestamp=timestamp,
        artifact_dir=artifact_dir,
        log_file=log_file,
    )


def setup_logging(log_file: Path) -> logging.Logger:
    """创建同时输出到终端和 UTF-8 文件的 logger。"""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 避免在同一个 Python 进程中重复调用时叠加 handler。
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # 显式指定 encoding="utf-8"，避免 Windows 默认编码造成中文乱码。
    file_handler = logging.FileHandler(
        log_file,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    """设置随机种子，并限制默认 CPU 线程数量。

    某些机器会给 PyTorch 分配很多 CPU 线程。
    对本项目这种中小型 Transformer，线程过多反而可能更慢。
    可通过环境变量 XIAOMA_TORCH_THREADS 手动覆盖。
    """

    thread_count = max(
        1,
        int(os.environ.get("XIAOMA_TORCH_THREADS", "1")),
    )
    torch.set_num_threads(thread_count)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch 只允许在并行任务开始前设置 inter-op 线程。
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    """根据命令行参数选择 CPU 或 CUDA 设备。"""

    requested = requested.lower()

    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA device is available."
        )

    return torch.device(requested)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """以 UTF-8、保留中文的方式写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def latest_run_dir(model_root: Path, branch: str) -> Path:
    """返回指定分支下按时间排序的最新模型目录。"""

    branch_root = model_root / safe_name(branch)
    candidates = sorted(
        (path for path in branch_root.glob("*") if path.is_dir()),
        key=lambda path: path.name,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No trained model was found under {branch_root}."
        )

    return candidates[-1]


def distance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """计算室内定位常用的二维欧氏距离误差。"""

    distances = np.linalg.norm(y_true - y_pred, axis=1)
    errors = y_true - y_pred

    return {
        "distance_mean": float(np.mean(distances)),
        "distance_median": float(np.median(distances)),
        "distance_p90": float(np.percentile(distances, 90)),
        "distance_max": float(np.max(distances)),
        "longitude_rmse": float(
            np.sqrt(np.mean(errors[:, 0] ** 2))
        ),
        "latitude_rmse": float(
            np.sqrt(np.mean(errors[:, 1] ** 2))
        ),
    }
