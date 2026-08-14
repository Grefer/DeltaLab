"""全局测试夹具。

这里只放**必须在每个测试上自动生效**的隔离。需要显式声明的夹具留在各测试
文件里（例如 ``history_store_dir``）。
"""
from __future__ import annotations

import pytest

import backtest_pool_store
import history_bar_cache
import history_store


@pytest.fixture(autouse=True)
def isolate_backtest_pool(tmp_path, monkeypatch):
    """把回测结果池的落盘目录指向临时目录。

    必须是 autouse：结果池是在 ``BacktestApp.__init__`` 里载入、在「保留结果」
    时写盘的，而测试里有二十来处直接构造真实 ``BacktestApp()``。少一处显式
    声明，那个测试就会读写开发者本机的 ``data/backtest_pool``——表现为测试
    之间互相看见对方的结果、条数断言随本地历史漂移，还会把结果包写进仓库。
    （这条夹具加进来之前，跑一次全量测试确实往仓库里落了 4 个包。）

    只替换 ``pool_dir``，不动 ``write_snapshot``/``read_all`` 的默认参数——
    它们内部都以 ``pool_dir()`` 为默认目录，替一处即可覆盖全链路。
    """
    pool = tmp_path / "backtest_pool"
    monkeypatch.setattr(
        backtest_pool_store, "pool_dir", lambda: str(pool))
    return pool


@pytest.fixture(autouse=True)
def isolate_history_bar_cache(tmp_path, monkeypatch):
    """把 bar 级明细缓存目录指向临时目录。

    同样必须是 autouse，理由和结果池一样但更隐蔽：写盘发生在「保留结果」和
    「策略优选跑完」这两条主流程里，测试并不显式碰缓存，却会顺带往开发者本
    机的 ``data/cache/history_bars`` 里灌几十兆——而且那些条目是内容寻址
    的，会被之后的真实运行命中，等于让测试数据流进生产结果。
    """
    cache = tmp_path / "history_bars"
    monkeypatch.setattr(
        history_bar_cache, "cache_dir", lambda: str(cache))
    return cache


@pytest.fixture(autouse=True)
def isolate_history_results(tmp_path, monkeypatch):
    """把策略优选结果包的落盘目录指向临时目录。

    这一条是补上的，补之前 ``data/history_results`` 里那份真实结果包消失
    了——查不出是哪个进程删的，但没有 autouse 隔离这件事本身就是漏洞：
    ``save_result`` 每次都会顺带 ``enforce_limit()``，只要有测试在真实目录
    里存够 20 份，用户自己存的那份就会被当成最旧的淘汰掉，而且不留痕迹。

    ``test_gui_strategy_state.py`` 里那个显式的 ``history_store_dir`` 夹具
    保留：它还负责把 tmp_path 交给测试用。两者指向同一个属性，谁后设都是
    临时目录，不冲突。
    """
    results = tmp_path / "history_results"
    monkeypatch.setattr(
        history_store, "results_dir", lambda: str(results))
    return results
