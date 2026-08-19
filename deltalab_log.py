# _*_ coding: utf-8 _*_
"""运行期日志：落盘、按大小滚动，冻结包里也能拿到现场。

**为什么需要它。** 打包出来的 `.app` 是 ``console=False``（deltalab.spec），
而运行期诊断此前全靠 ``print(..., file=sys.stderr)``——在冻结包里
``sys.stderr is None``，那句话是个静默 no-op：不报错，也什么都不写。于是
用户说"我这儿特别慢"或"点了没反应"时，现场零信息，只能靠猜。

**写到哪。** 与结果池、逐 bar 缓存同一套约定：冻结包写用户目录，开发时写
仓库内的 ``data/``。日志按 2 MB 滚动、留 3 份，占用封顶 8 MB——它是给人看
的现场记录，不是审计流水，不该无限长。

**绝不能因为日志把程序拖垮。** 目录不可写、磁盘满、权限不对——任何一种都
只让日志退化成"什么都不记"，主流程照常跑。这也是为什么 setup() 里那圈
except 是宽的：写日志失败本身没有任何值得中断用户操作的理由。
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys

LOGGER_NAME = "deltalab"

# 单个文件 2 MB × 4 份（当前 + 3 个备份）= 封顶 8 MB。
MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 3

_configured = False


def log_dir() -> str:
    """与 history_bar_cache.cache_dir / history_store.results_dir 同一套约定。"""
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.expanduser("~"), ".deltalab", "logs")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "logs")


def log_path() -> str:
    return os.path.join(log_dir(), "deltalab.log")


def get_logger(name: str | None = None) -> logging.Logger:
    """取一个子 logger。模块级直接调用即可，不必先 setup。

    没 setup 过时它照样能用——只是没有落盘处理器，等于什么都不写。这样
    ``pytest`` 跑纯逻辑时不会凭空在仓库里造出日志文件。
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def setup(level: int = logging.INFO, *, to_stderr: bool | None = None) -> str | None:
    """装上落盘处理器，返回日志文件路径；装不上就返回 None。

    幂等：重复调用不会叠加处理器（GUI 入口与测试都可能各调一次）。

    ``to_stderr`` 默认跟随 ``sys.stderr`` 是否可用——开发时同时打到终端，
    冻结包里没有终端就只落盘。
    """
    global _configured
    logger = get_logger()
    logger.setLevel(level)
    # 不往 root 冒泡：本包的日志只走自己的处理器，免得宿主程序（或 pytest
    # 的 caplog）被灌进一堆本不属于它的记录。
    logger.propagate = False
    if _configured:
        return getattr(logger, "_deltalab_path", None)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    path = None
    try:
        os.makedirs(log_dir(), exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_path(), maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
            encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        path = log_path()
    except Exception:                                  # noqa: BLE001
        # 目录不可写 / 磁盘满 / 权限不对：日志退化成什么都不记，主流程照跑。
        # 这里绝不能抛——为了写日志把程序打断是本末倒置。
        path = None

    if to_stderr is None:
        to_stderr = sys.stderr is not None
    if to_stderr:
        try:
            stream = logging.StreamHandler()
            stream.setFormatter(formatter)
            logger.addHandler(stream)
        except Exception:                              # noqa: BLE001
            pass

    logger._deltalab_path = path                       # type: ignore[attr-defined]
    _configured = True
    return path


def describe_target() -> str:
    """给界面用的一句话：日志写到哪儿了。"""
    logger = get_logger()
    path = getattr(logger, "_deltalab_path", None)
    return path or "（日志未启用：目录不可写）"
