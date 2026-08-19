# _*_ coding: utf-8 _*_
"""
期权定价基类，提供通用的有限差分 Greeks 计算方法
"""

import contextlib
import copy
import datetime
import hashlib
import struct
import threading

import numpy as np

try:
    from .constants import ANNUAL_DAYS
except ImportError:
    from constants import ANNUAL_DAYS


# ---- bar 级定价记忆化 ----------------------------------------------------
#
# 一次策略优选里，同一段行情上的每个候选策略都会各自跑一遍 HedgeBacktest，
# 而**定价与策略无关**：第 i 根 bar 的定价对象状态只是
# ``(option_init, S[0..i], steps_per_day, timestamps)`` 的纯函数，策略只决定
# 哪些 bar 要额外调 get_greeks。于是 8 个候选把同一批价格算了 8 遍。
# 叠加 get_greeks 内部 10 次 bump 里 price0 与该 bar 自身定价同状态，
# 实测 GUI 默认配置下 4224 次定价里只有 640 个互不相同。
#
# 这里用一个作用域内的 dict 把它们记下来。**不改变任何数值**——只是把
# "重算一遍同一个纯函数"换成"读上次的结果"。
#
# 载体是 threading.local：作用域由调用方用 price_memo() 显式开启，没开启
# 时 priced() 直接透传 get_price()，因此 run_multi 那种每线程独立跑的路径
# 不受影响。若将来要做跨线程的 bar 级预热，得把 dict 改成显式传递——
# thread-local 下预热线程只会写进自己那份，主线程一条也读不到。
_MEMO = threading.local()

# 单个作用域最多记多少条。超出后不再新增，已有的照常命中——纯粹是内存
# 保险丝，不影响任何数值。上限按"最坏情况约 60 MB"定：每条约 32 字节
# 摘要 + dict 槽位 + 一个 float，实测约 150 字节/条。
# 日频一段最多几千条，够用几十倍；1 分钟粒度的长段会在这里封顶，代价是
# 后面的 bar 各候选各算一次，回到改动前的行为。
MEMO_MAX_ENTRIES = 400_000


@contextlib.contextmanager
def price_memo():
    """在该作用域内记忆化 ``priced()`` 的结果。

    可嵌套：退出时恢复外层的 dict。作用域结束即释放，缓存不跨段留存。
    """
    prev = getattr(_MEMO, "d", None)
    _MEMO.d = {}
    try:
        yield _MEMO.d
    finally:
        _MEMO.d = prev


def _feed(hasher, value, depth=0):
    """把属性值喂进摘要器；认不出的类型返回 False 表示放弃缓存。

    白名单式：不认识就不缓存，绝不猜一个 hash。默认的 ``object.__hash__``
    按身份算，原地改过的可变对象会算出同一个 key——那是读到过期价格的方向，
    比不缓存坏得多。

    每一项都带类型标签和长度前缀：裸值拼接时 ``("ab", "c")`` 和
    ``("a", "bc")`` 会喂出同一串字节，而 1 / 1.0 / True 三者互相 ``==``、
    哈希也相同。打上标签之后最坏只是多算一次（miss），不会拿错价格。
    """
    if depth > 6:
        return False
    if value is None:
        hasher.update(b"N;")
        return True
    # bool 必须排在 int 前面——它是 int 的子类。
    if isinstance(value, (bool, np.bool_)):
        hasher.update(b"b1;" if value else b"b0;")
        return True
    if isinstance(value, (int, np.integer)):
        hasher.update(b"i%d;" % int(value))
        return True
    if isinstance(value, (float, np.floating)):
        hasher.update(b"f")
        hasher.update(struct.pack("<d", float(value)))
        hasher.update(b";")
        return True
    if isinstance(value, (str, bytes)):
        raw = value.encode("utf-8") if isinstance(value, str) else value
        hasher.update(b"s%d:" % len(raw))
        hasher.update(raw)
        hasher.update(b";")
        return True
    if isinstance(value, (datetime.date, datetime.time, datetime.datetime)):
        raw = value.isoformat().encode("ascii")
        hasher.update(b"t%d:" % len(raw))
        hasher.update(raw)
        hasher.update(b";")
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return False
        raw = np.ascontiguousarray(value).tobytes()
        hasher.update(b"a%s%s%d:" % (
            str(value.shape).encode("ascii"),
            value.dtype.str.encode("ascii"), len(raw)))
        hasher.update(raw)
        hasher.update(b";")
        return True
    if isinstance(value, (list, tuple)):
        # 全数值容器直接按 float64 字节喂。observ / sr / ko_observ 动辄几百
        # 上千个元素，逐元素走通用分支既慢又没必要。
        # bool 不走这条：它和 1 会压成同一个 float64，得留给上面的带标签分支。
        if _all_plain_numbers(value):
            try:
                arr = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                arr = None
            if arr is not None and arr.ndim == 1:
                raw = arr.tobytes()
                hasher.update(b"q%d:" % len(raw))
                hasher.update(raw)
                hasher.update(b";")
                return True
        hasher.update(b"[%d:" % len(value))
        for item in value:
            if not _feed(hasher, item, depth + 1):
                return False
        hasher.update(b"];")
        return True
    return False


def _all_plain_numbers(seq):
    """序列是否全是数值。空序列算是——它没有元素可分类，而两条分支必须
    一路等价到长度 0，否则空 ``sr`` 会因走向不同分支而算出两个 key。
    """
    return all(
        isinstance(item, (int, float, np.integer, np.floating))
        and not isinstance(item, (bool, np.bool_))
        for item in seq)


def _memo_key(obj):
    """枚举全部公开非可调用属性，摘成定价身份；有一项摘不出就返回 None。

    全枚举而不是挑白名单：``optiontype`` 是 Option_AB/DE/SNB 的定价方法分派
    键（``getattr(self, self.optiontype)()``），漏掉它会让 13 个累计子类型
    算出同一个 key 并互相读到对方的价格。``sr``/``ko_observ``/``observ``
    同理。这条教训在 history_bar_cache._digest_object 的注释里已经付过一次
    学费，这里不再重犯。同样的理由，这里**不留跳过名单**——留一个口子就是
    留一次静默错价的机会。

    私有属性只显式补 ``_intraday_elapsed``：它是唯一进入定价的私有字段
    （Option_Vanilla 在 get_price 里消费），其余 ``_`` 开头的都是属性方法。
    tests/test_pricing_memo.py 有一条守卫测试盯着这个前提。

    **返回定长摘要而不是原始元组。** 元组 key 在 1 分钟粒度下会失控：
    T=66、spd=240 时每个 key 约 4.6 KiB，最坏 17 万条就是 700 MB 以上。
    32 字节摘要把它压成常数。代价是理论上的碰撞——256 位下概率可以忽略，
    且 history_bar_cache 的磁盘 key 早就是同一套做法（sha1 内容摘要）。
    """
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(type(obj).__name__.encode("utf-8"))
    hasher.update(b";")
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:                              # noqa: BLE001
            return None
        if callable(value):
            continue
        hasher.update(b"%d:" % len(name))
        hasher.update(name.encode("utf-8"))
        hasher.update(b"=")
        if not _feed(hasher, value):
            return None
    hasher.update(b"_intraday_elapsed=")
    hasher.update(struct.pack("<d", float(obj._intraday_elapsed)))
    return hasher.digest()


class OptionBase:
    """期权定价基类

    子类需实现：
        - get_price() -> float
        - _time_remaining (property) -> float/int : 剩余到期时间（交易日数）
        - _theta_overrides(dt) -> dict : theta bump 所需的参数覆盖
        - _decrement_time() : 时间推进一天

    MC Greeks 说明：
        对 MC 定价的子类（Option_AB / Option_AS / Option_DE），get_greeks
        在有限差分 bump-and-reprice 过程中会复用 self.mc_seed 以实现
        Common Random Numbers (CRN)：所有 bump 路径共享同一组随机数，
        以保证不同 bump 的定价差由真实参数敏感性产生，而不是 MC 噪声。
        解析定价的 Option_Vanilla 不受这条约束。
    """

    # 日内已流逝比例（0~1，单位=日）。bump copy 里用它把 T 方向的剩余时间
    # 按小数日衰减；默认 0 不影响现有行为。目前仅 Option_Vanilla 在 get_price
    # 里消费这个字段，MC 类的 T_days 需要整数 nStep，暂不支持 intraday 衰减。
    _intraday_elapsed = 0.0

    # MC 采样种子：子类实例可单独覆盖。None = 每次调用 McGbmQ 都用 OS 熵
    # 重新抽样；int = 确定性序列，bump 之间共享即为 CRN。默认 20 保持
    # 与 Review 前行为一致（所有 MC 期权共用一组路径）。
    mc_seed = 20

    # Greeks bump 时使用的路径数；None = 与 self.nPath 一致。某些 MC 期权
    # 路径很重，bump 五个方向（±ds, ±2ds, ±sigma, ±r, theta）可能明显变慢，
    # 这里允许外部覆盖为更小值（例如 max(nPath//4, 5000)），以换取性能。
    greeks_nPath = None

    # 可以在构造期覆盖的基类级选项。它们是 OptionBase 的**类属性**，不是各
    # 子类 __init__ 的形参，所以 ``Option_DE(..., mc_seed=7)`` 会掉进
    # ``**kwargs`` 里被静默吞掉——不报错、也不生效，所有实例继续共用同一批
    # 随机数。写"不同 seed 跑多次取标准差"的脚本最容易在这里翻车：标准差
    # 恒为 0，看着像是定价稳得离谱。
    _BASE_OVERRIDES = ("mc_seed", "greeks_nPath")

    def _apply_extra_options(self, options):
        """消化各子类 ``__init__`` 的 ``**kwargs``。

        认识的落到实例上，不认识的报错——这个 ``**kwargs`` 此前从未被读过，
        纯粹是个静默丢弃口，连参数名拼错都不会有任何提示。用 ``TypeError``
        是为了与 Python 自己对多余关键字参数的反应一致。
        """
        for name in self._BASE_OVERRIDES:
            if name in options:
                setattr(self, name, options.pop(name))
        if options:
            raise TypeError(
                f"{type(self).__name__} 收到无法识别的参数: "
                + "、".join(sorted(options)))

    @property
    def _time_remaining(self):
        raise NotImplementedError

    def get_price(self):
        raise NotImplementedError

    def _theta_overrides(self, dt):
        raise NotImplementedError

    def _decrement_time(self):
        """推进一天的时间参数调整，由子类实现"""
        raise NotImplementedError

    def step_forward(self, new_price):
        """推进一个交易日：当前 s0 进入历史记录，更新为新价格，时间减一天"""
        if hasattr(self, 'sr'):
            self.sr = list(self.sr) + [self.s0]
        self.s0 = new_price
        self._decrement_time()
        # 跨日后重置日内 elapsed，避免 bump copy 继承残值
        self._intraday_elapsed = 0.0

    def knockout_event(self, prices, steps_per_day=1):
        """提前了结（敲出）检测钩子，供对冲回测在已知价格路径时截断存续期。

        默认返回 None（普通期权不会提前了结）。路径依赖且带敲出观察的结构
        （如雪球 Option_SNB）可重写：给定回测价格路径 prices（prices[0]=今日）
        与 steps_per_day，返回 (i_ko, settle_value)——i_ko 为 prices 中触发敲出
        的索引，settle_value 为敲出当日的结算价值（之后不再有现金流）。
        """
        return None

    def priced(self):
        """带作用域记忆化的 ``get_price()``；没开作用域时就是它本身。

        只在 ``price_memo()`` 作用域内生效。子类不需要（也不应该）重写它——
        重写 ``get_price`` 即可，记忆化在这一层统一处理。
        """
        memo = getattr(_MEMO, "d", None)
        if memo is None:
            return self.get_price()
        # mc_seed=None 的语义是"每次调用 McGbmQ 都用 OS 熵重新抽样"
        # （见本文件顶部 mc_seed 的说明）。缓存会让两次调用返回同一个值，
        # 把这条契约静默改掉，所以这里必须绕开。
        if getattr(self, "mc_seed", 20) is None:
            return self.get_price()
        key = _memo_key(self)
        if key is None:                 # 摘要不出来就不缓存，行为退回原样
            return self.get_price()
        if key in memo:
            return memo[key]
        value = self.get_price()
        if len(memo) < MEMO_MAX_ENTRIES:
            memo[key] = value
        return value

    def _bumped_copy(self, **overrides):
        """创建参数副本用于 bump-and-reprice，不修改原对象状态。

        父子对象不能共享任何可变容器：get_greeks 里 _theta_overrides 会对
        sr 做 list(...) + [s0] 新建列表，但 gamma / delta bump 路径只改 s0，
        不新建 sr；若是裸浅拷贝，后续若有任何分支原地 append 到 self.sr，
        父子对象会相互污染。

        这里用"浅拷贝 + 逐个复制可变容器"代替 deepcopy：容器里装的都是
        不可变标量（float / int / date / str），复制一层就够，不必递归。
        实测比 deepcopy 快 3.4~46 倍，且从 O(len(sr)) 变成 O(1)——
        deepcopy 的成本随已实现前缀 sr 线性增长，正好在回测后半程最贵。
        """
        obj = copy.copy(self)
        d = obj.__dict__
        for k, v in list(d.items()):
            t = type(v)
            if t is list:
                d[k] = list(v)
            elif t is dict:
                d[k] = dict(v)
            elif t is set:
                d[k] = set(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.copy()
        for k, v in overrides.items():
            setattr(obj, k, v)
        return obj

    @staticmethod
    def _safe_price(val):
        """将 None 返回值转为 0.0"""
        return val if val is not None else 0.0

    def get_greeks(self):
        """有限差分法计算 Greeks: [delta, gamma, vega, theta, rho]

        对 MC 期权（self 有 nPath 属性时），在 bump 期间临时将 self.nPath
        切换为 self.greeks_nPath（若已设置），并保持 self.mc_seed 不变，
        以实现 CRN：所有 bump 共享同一批随机数路径，bump 定价差仅反映
        参数敏感性而非 MC 噪声。Option_Vanilla 走解析路径，不受影响。
        """

        if self._time_remaining <= 0:
            return [0, 0, 0, 0, 0]

        ds = min(20.0, self.s0 / 100.0)
        sigma0 = float(self.sigma)
        dz = max(0.005, abs(sigma0) * 0.01)
        dt = 1
        dr = 0.01

        # ---- MC 期权：bump 前临时压路径数，bump 结束后恢复 ----
        _npath_saved = None
        if hasattr(self, "nPath") and getattr(self, "greeks_nPath", None):
            _npath_saved = self.nPath
            self.nPath = int(self.greeks_nPath)

        try:
            price0 = self._safe_price(self.priced())

            # delta（中心差分）
            price_up = self._safe_price(self._bumped_copy(s0=self.s0 + ds).priced())
            price_dn = self._safe_price(self._bumped_copy(s0=self.s0 - ds).priced())
            delta = (price_up - price_dn) / (2 * ds)

            # gamma（二阶中心差分，步长 2*ds）
            price_up2 = self._safe_price(self._bumped_copy(s0=self.s0 + 2 * ds).priced())
            price_dn2 = self._safe_price(self._bumped_copy(s0=self.s0 - 2 * ds).priced())
            gamma = (price_up2 - 2 * price0 + price_dn2) / (4 * ds ** 2)

            # vega
            sigma_up = sigma0 + dz
            sigma_dn = max(sigma0 - dz, 1e-8)
            price_vup = self._safe_price(self._bumped_copy(sigma=sigma_up).priced())
            price_vdn = self._safe_price(self._bumped_copy(sigma=sigma_dn).priced())
            vega = (price_vup - price_vdn) / (sigma_up - sigma_dn)

            # theta
            price_theta = self._safe_price(self._bumped_copy(**self._theta_overrides(dt)).priced())
            theta = (price_theta - price0) / (dt / ANNUAL_DAYS)

            # rho
            price_rup = self._safe_price(self._bumped_copy(r=self.r + dr).priced())
            price_rdn = self._safe_price(self._bumped_copy(r=self.r - dr).priced())
            rho = (price_rup - price_rdn) / (2 * dr)
        finally:
            if _npath_saved is not None:
                self.nPath = _npath_saved

        return [delta, gamma, vega, theta, rho]
