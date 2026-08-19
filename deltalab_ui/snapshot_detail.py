# _*_ coding: utf-8 _*_
"""快照签名的摊平、格式化、明细分节与多结果差异比对。

结果池的「参数详情」和「结果对比」两个页面，以及策略优选的参数详情，最终都
落到这一层：把 ``market_key`` / ``contract_key`` / ``economics_key`` 三组签名
元组摊成 ``{字段: 值}``，按显示名排版成分节，再算出多份快照之间**差在哪一
项**。三处必须共用同一份实现——各写一份的话，加一个字段只同步一边，同一次回
测在两个窗口会显示出不一样的输入。

全是纯函数，输入是快照对象或 state dict，不碰 tkinter。调用方仍写
``BacktestApp._snapshot_detail_sections(...)``，类里按同名 staticmethod 别名
暴露，与 ``history_selection`` / ``wind_resolve`` 两组一致。
"""

import copy
import datetime
import hashlib
from types import SimpleNamespace

import numpy as np

import history_selection
from deltalab_ui import wind_resolve
from deltalab_ui.constants import (
    OPTION_CLASSES,
    SIGMA_SOURCE_DISPLAY,
    STRATEGY_DISPLAY,
    SUBTYPE_DISPLAY,
    WIND_AUTO_BAR_SIZE,
)


def freeze_snapshot_value(value):
    """把配置规范化为可稳定比较的不可变值。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return tuple(sorted(
            (str(key), freeze_snapshot_value(item))
            for key, item in value.items()
        ))
    if isinstance(value, (list, tuple)):
        return tuple(freeze_snapshot_value(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        return ("ndarray", str(array.dtype), tuple(array.shape), digest)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def saved_snapshot_position(snapshot):
    """读取快照的经济方向；显示名称和其它参数均不能覆盖它。"""
    return wind_resolve.normalize_position(snapshot.position)


def signature_keys_from_state(gui_state, *, data_digest=None,
                               steps_per_day=None):
    """从一份 gui_state 造出行情 / 期权 / 规模与成本三组签名。

    单独抽出来是因为**策略优选的参数详情也要用它**：那边只有一份运行时
    冻结的 state、没有 bt，但要摊出的字段与结果池详情必须逐项对齐。留在
    ``_make_saved_backtest_result`` 里的话两处各写一份字典，加一个字段只
    同步一边，同一次回测在两个窗口就会显示出不一样的输入。
    """
    return (
        freeze_snapshot_value({
            "source": gui_state.get("source"),
            # 价格序列本身的摘要。可读字段（标的、区间、粒度）说不清的
            # 差异由它兜底：日内滚动的两段、同一区间取到了不同数据，都能
            # 认出来。反过来，跨周期取到同一段末尾时数据逐字节相同，它也
            # 不会造出假差异——用分段编号就会，那三段编号不同而数据一模
            # 一样。优选那边跑的是多段，没有单一摘要，传 None。
            "data_digest": data_digest,
            "seed": gui_state.get("seed"),
            "csv_path": gui_state.get("csv_path"),
            "csv_col": gui_state.get("csv_col"),
            "wind_code": gui_state.get("wind_code"),
            "wind_start": gui_state.get("wind_start"),
            "wind_end": gui_state.get("wind_end"),
            "wind_bar_size": gui_state.get("wind_bar_size"),
        }),
        freeze_snapshot_value({
            "cls_name": gui_state.get("cls_name"),
            "subtype": gui_state.get("subtype"),
            "params": gui_state.get("params", {}),
        }),
        freeze_snapshot_value({
            "quantity": gui_state.get("quantity"),
            "multiplier": gui_state.get("multiplier"),
            "tc_rate": gui_state.get("tc_rate"),
            "slippage_bps": gui_state.get("slippage_bps", 0.0),
            "steps_per_day": steps_per_day,
        }),
    )


def snapshot_source(snapshot):
    """快照的行情来源；取不到时 ``None``。"""
    flat = flatten_signature(
        tuple(getattr(snapshot, "market_key", ()) or ()))
    source = flat.get(("source",))
    return source if isinstance(source, str) else None


# 每种对冲策略实际读取的 form_state 字段。用于对比签名——回填不需要它，
# 回填要的是"把表单恢复原样"，全量记录对回填是正确的。
_STRATEGY_RELEVANT_FORM_FIELDS = {
    "close_to_close": (),
    "fixed_times": ("fixed_times",),
    "hedge_band": (
        "interval_type", "price_interval", "sigma_source", "sigma_window"),
}


# 签名里恒定出现的策略参数字段（无关时填 None，保持键集合稳定）。
_STRATEGY_SIGNATURE_FIELDS = tuple(dict.fromkeys(
    field
    for fields in _STRATEGY_RELEVANT_FORM_FIELDS.values()
    for field in fields
))


def snapshot_strategy_signature(snapshot):
    """对比用的对冲策略签名：只含**这个策略真正读取**的字段。

    不能拿整份 ``form_state`` 当签名。``_snapshot_form_state`` 是为回填
    表单服务的，它不看 ``strategy_name`` 就全量记录固定时刻与带宽三项；
    而 ``_collect_history_state`` 每轮优选都会把本轮候选配置写回 state，
    重放快照因此带着与自己无关的残值。结果是两条**跑的是同一策略、逐日
    损益完全相同**的结果被报成「本次对比的变量：对冲策略」，还被染成绿色
    （干净的单变量实验）——例如两轮优选只有固定时刻候选文本不同，都选
    每日收盘基线、加载同一段，close_to_close 根本不读 fixed_times。

    未知策略名保守起见回退到全量比较：宁可多报一次差异，也不要把真的
    不同说成相同。

    **键集合必须恒定**，无关字段填 ``None`` 而不是删掉。``
    differing_field_names`` 在两侧键集合不同时会整体放弃字段级定位、
    只报到属性级；若这里按策略删键，「每日收盘 vs 固定间隔」这种正当的
    跨策略对比就再也说不出「策略类型 / 带宽单位 / 带宽阈值变了」，退化
    成一句光秃秃的「对冲策略」。填 ``None`` 则两边仍可逐字段比：同策略
    时无关项两边都是 ``None`` 不产生假差异，跨策略时 ``None`` 与实际值
    不同，照常点名。
    """
    state = dict(getattr(snapshot, "form_state", None) or {})
    name = str(state.get("strategy_name", "") or "")
    relevant = _STRATEGY_RELEVANT_FORM_FIELDS.get(name)
    if relevant is None:
        return freeze_snapshot_value(state)
    # 策略名与收盘保底对所有策略都生效，恒参与比较。
    signature = {
        "strategy_name": name,
        "force_day_close_hedge": bool(
            state.get("force_day_close_hedge", False)),
    }
    for key in _STRATEGY_SIGNATURE_FIELDS:
        signature[key] = state.get(key) if key in relevant else None
    return freeze_snapshot_value(signature)


# 一次对比里可以变化的五组属性。控制变量的道理：只有一组不同，差异才
# 归得到那一组头上；同时变两组以上，看到的差就说不清是谁造成的。
# (属性键, 中文名, 取值函数)
_COMPARISON_ASPECTS = (
    ("market", "行情", lambda s: getattr(s, "market_key", ())),
    # 这一组统辖的是「是哪一种期权」外加它的全部合约参数，组名必须把两
    # 件事都说到：只写「期权类型」的话，改了执行价、类型没动的那次对比
    # 会被报成「本次对比的变量：期权类型（执行价：100 vs 105）」——句子
    # 本身就在说类型变了。
    ("contract", "期权类型与参数",
     lambda s: getattr(s, "contract_key", ())),
    ("position", "头寸方向", lambda s: (
        ("position", saved_snapshot_position(s)),)),
    ("economics", "规模与成本", lambda s: getattr(s, "economics_key", ())),
    ("strategy", "对冲策略",
     lambda s: snapshot_strategy_signature(s)),
)


# 字段级差异的中文名。签名保留了键名，所以能说到具体是哪一项不同，
# 而不只是"期权不同"。期权合约参数的标签直接取自 OPTION_CLASSES 的
# 定义，新增期权类型时不必在这里补一遍。
_COMPARISON_FIELD_LABELS = {
    "source": "行情来源", "seed": "随机种子", "csv_path": "CSV 文件",
    "csv_col": "CSV 列", "wind_code": "标的代码",
    "wind_start": "起始日", "wind_end": "截止日",
    "wind_bar_size": "bar 粒度", "data_digest": "行情数据",
    "cls_name": "期权大类", "subtype": "期权类型",
    "quantity": "交易数量", "multiplier": "合约乘数",
    "tc_rate": "成本率", "slippage_bps": "滑点", "steps_per_day": "日内采样",
    "position": "头寸方向",
    "strategy_name": "策略类型", "fixed_times": "固定时刻",
    "interval_type": "带宽单位", "price_interval": "带宽阈值",
    "force_day_close_hedge": "收盘兜底",
    "sigma_source": "波动率口径", "sigma_window": "波动率窗口",
}


def option_param_labels(cls_name=None):
    """期权合约参数的键 → 中文名，以及取值的中文回译。

    标签就写在 OPTION_CLASSES 的参数定义里（第二项），带 choices 的参数
    （如方向 1/-1）还能把数值译回"看涨/看跌"。

    **给了大类就以那个大类自己的定义为准。** 同一个键在不同大类下是不同
    的东西，全局合并会张冠李戴：

    - ``N`` 在累计里是杠杆倍数，在亚式里是观察日数；
    - ``T_days`` 在香草里是期限，在累计里是剩余期限；
    - ``s0`` 在雪球里是最新价而非初始价格；
    - ``cp`` 最狠——雪球的 ``-1`` 是「雪球 (卖看跌)」、``+1`` 是「反雪球
      (卖看涨)」，其余各类是「看跌 / 看涨」。译错的不是名字而是取值本身。

    此前一律 ``setdefault`` 且按 OPTION_CLASSES 的定义序合并，等于永远
    用香草那一套去解释所有期权。

    实现是把目标大类排到最前面再合并：它的定义先落进 setdefault 因而胜
    出，其余大类留作回退，未在本类定义的键仍能拿到一个名字而不是裸键名。
    拿不到大类时（跨大类比对）行为与从前一致——那时两侧键集合本来就对不
    上，差异只报到属性级，标签取谁都不影响结论。
    """
    configs = [OPTION_CLASSES[cls_name]] if cls_name in OPTION_CLASSES else []
    configs.extend(
        config for name, config in OPTION_CLASSES.items()
        if name != cls_name)
    labels, choices = {}, {}
    for config in configs:
        for spec in config.get("params", ()):
            key = str(spec[0])
            labels.setdefault(key, str(spec[1]))
            if len(spec) > 4 and isinstance(spec[4], dict):
                choices.setdefault(key, {
                    value: text for text, value in spec[4].items()})
    return labels, choices


def signature_cls_name(flats):
    """从若干摊平签名里取共同的期权大类；不一致或没有则 ``None``。

    标签与取值回译要按大类取（见 ``option_param_labels``），而
    ``differing_field_names`` 是五组属性共用的——不含 ``cls_name`` 的那
    几组自然拿到 ``None``，正好退回全局合并。
    """
    names = {flat.get(("cls_name",)) for flat in flats}
    if len(names) != 1:
        return None
    name = names.pop()
    return name if isinstance(name, str) else None


def flatten_signature(value, prefix=()):
    """把嵌套签名摊平成 {键路径: 取值}。

    ``contract_key`` 里的 params 本身又是一组键值对，只比顶层就只能说
    到"合约参数不同"——而真正要说的是"波动率 0.18 vs 0.15"。
    """
    flat = {}
    for key, item in value:
        path = prefix + (str(key),)
        nested = (
            isinstance(item, tuple) and item
            and all(isinstance(entry, tuple) and len(entry) == 2
                    for entry in item))
        if nested:
            flat.update(flatten_signature(item, path))
        else:
            flat[path] = item
    return flat


# 取值的中文回译：差异串里写 "close_to_close vs hedge_band" 没人看得舒服。
_COMPARISON_VALUE_LABELS = {
    "position": {-1: "买入", 1: "卖出"},
    "strategy_name": STRATEGY_DISPLAY,
    "sigma_source": SIGMA_SOURCE_DISPLAY,
    "subtype": SUBTYPE_DISPLAY,
    "interval_type": {
        "absolute": "绝对", "relative": "相对", "sigma": "σ 倍数"},
    "force_day_close_hedge": {True: "开启", False: "关闭"},
    "source": {"simulate": "模拟", "csv": "CSV", "wind": "Wind"},
}


def format_signature_value(key, value, cls_name=None):
    """把签名里的取值渲染成人看的字符串。

    ``cls_name`` 决定用哪一套合约参数回译：不给的话 ``cp=-1`` 在雪球上会
    被译成「看跌 (Put)」，而雪球的 ``-1`` 是「雪球 (卖看跌)」。
    """
    mapping = _COMPARISON_VALUE_LABELS.get(key)
    if mapping and value in mapping:
        return str(mapping[value])
    _labels, choices = option_param_labels(cls_name)
    mapping = choices.get(key)
    if mapping and value in mapping:
        return str(mapping[value])
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "开启" if value else "关闭"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def differing_field_names(values):
    """逐字段比出差异，返回 [(中文名, [各条的取值]), ...]。

    取值按传入顺序原样返回，不拼成 "A vs B" ——三条以上时那种串既看不
    出谁是谁，又会把标题撑到七八十字。拼不拼、怎么排版交给展示层。
    不可读的字段（价格序列摘要）取值给 None，表示"只说不同、无值可看"。

    键集合本身不同时（换期权大类会整体换掉参数集）说不出具体字段，返回
    空表示"整组不同"，由调用方按属性级报告。
    """
    flats = [flatten_signature(value) for value in values]
    paths = {frozenset(flat) for flat in flats}
    if len(paths) != 1:
        return []
    cls_name = signature_cls_name(flats)
    param_labels, _choices = option_param_labels(cls_name)
    order = list(_COMPARISON_FIELD_LABELS) + list(param_labels)
    changed = []
    opaque = {"data_digest"}
    for path in next(iter(paths)):
        taken = [flat[path] for flat in flats]
        if len({freeze_snapshot_value(v) for v in taken}) <= 1:
            continue
        key = path[-1]
        label = _COMPARISON_FIELD_LABELS.get(
            key, param_labels.get(key, key))
        rank = order.index(key) if key in order else len(order)
        if key in opaque:
            # 摘要是一串哈希，摊出来没人看得懂；它只负责"确实不一样"。
            changed.append((rank, label, None))
            continue
        changed.append((rank, label, [
            format_signature_value(key, value, cls_name)
            for value in taken
        ]))
    # 可读字段已经说清了差异时，就不必再补一句"行情数据不同"——那是
    # 同一件事的两种说法。只有摘要单独不同才值得说：输入看着一样，跑出
    # 来的价格序列却不是同一条。
    if any(shown is not None for _rank, _label, shown in changed):
        changed = [item for item in changed if item[2] is not None]
    # 按定义顺序输出，而不是签名的字典序——"策略类型、带宽阈值"读着顺。
    # 次级键是标签本身：路径集合是 frozenset，迭代序不确定，只按 rank
    # 排的话未列入定义表的字段每次出来的顺序都可能不同。
    changed.sort(key=lambda item: (item[0], item[1]))
    return [(label, shown) for _rank, label, shown in changed]


# 详情窗里不列的字段。行情数据摘要是一串 sha256：在差异串里它至少还回答
# 了"这两条确实不是同一段数据"，单看一条时一个哈希什么也没说。
_SNAPSHOT_DETAIL_HIDDEN_FIELDS = frozenset({"data_digest"})


# 行情组按来源只列本次真正用到的那几项。三种来源的字段互斥，但
# market_key 恒定记全部键（键集合恒定是差异比对的要求），未用到的那几项
# 存的是**左侧控件当时的值**——不是空值，所以"渲染成 — 就跳过"拦不住：
# 模拟跑出来的快照照样会列出「CSV 列 close」「标的代码 510050.SH」，那是
# Wind 代码框里恰好还留着的内容，这次回测根本没碰过它。
_MARKET_DETAIL_FIELDS = {
    "simulate": ("source", "seed"),
    "csv": ("source", "csv_path", "csv_col"),
    "wind": ("source", "wind_code", "wind_start", "wind_end",
             "wind_bar_size"),
}


def bar_size_detail_text(actual, requested):
    """bar 粒度的显示串：把「实际用了什么」和「是不是自动推的」都写上。

    快照的 ``wind_bar_size`` 存的已经是 ``_resolve_wind_bar_size`` 解析后
    的实际粒度（见 ``_resolve_wind_backtest_state``），下拉原值另存在
    ``bar_size_requested``。只显示实际值看不出这个「15分钟」是自己挑的还
    是程序按策略推出来的；只显示「自动（推荐）」更糟——那根本不是一个粒度，
    是个待解析的占位。
    """
    actual = str(actual or "").strip()
    requested = str(requested or "").strip()
    if requested != WIND_AUTO_BAR_SIZE:
        return actual or "—"
    if not actual or actual == WIND_AUTO_BAR_SIZE:
        # 本字段上线前保留的快照没记原值，也可能记的就是未解析的占位：
        # 诚实说没记，不要把「自动（推荐）」摆出来充当一个粒度。
        return f"{WIND_AUTO_BAR_SIZE} · 未记录实际粒度"
    return f"{actual}（自动推荐）"


def intraday_steps_detail_text(snapshot, declared):
    """日内采样的显示串：写引擎**实际**用的那个数。

    见 ``_effective_intraday_steps``——签名里的值对真实行情是占位 1。
    """
    effective = int(getattr(snapshot, "intraday_steps", 0) or 0)
    if effective > 0:
        return str(effective)
    if declared is None:
        # 优选的一行覆盖多段，没有单一采样密度可言：交回 "—" 让渲染层
        # 整条略过，而不是含糊地说一句"未记录"。
        return "—"
    if snapshot_source(snapshot) == "simulate":
        # 模拟的采样密度是用户直接填进去的，签名里那个值就是事实。
        return str(max(1, int(declared or 1)))
    # 真实行情的旧快照只记了占位值，说不出实际粒度就别假装说得出。
    return "未记录（旧快照存的是占位值）"


# 展示层要覆盖签名取值的那几项：签名记的是「当时传给引擎的输入」，而这
# 两项的输入是占位或待解析的，人要看的是「实际跑的是什么」。收成一张表
# 而不是在渲染循环里堆 if——第三项迟早会来。
_DETAIL_VALUE_OVERRIDES = {
    ("market", "wind_bar_size"): (
        lambda snapshot, value: bar_size_detail_text(
            value, getattr(snapshot, "bar_size_requested", ""))),
    ("economics", "steps_per_day"): (
        lambda snapshot, value: intraday_steps_detail_text(
            snapshot, value)),
}


def format_detail_value(key, value, cls_name=None):
    """详情窗的取值渲染：在差异串那套回译之上补一条多值参数的排版。

    签名里的数组被冻成 ``("ndarray", dtype, shape, 摘要)`` 四元组，普通
    列表冻成元组；两者直接 ``str()`` 出来都是 Python 字面量。
    """
    if isinstance(value, tuple):
        if not value:
            return "—"
        if value[0] == "ndarray" and len(value) == 4:
            count = int(np.prod(value[2])) if value[2] else 1
            return f"数组（{count} 项）"
        return "、".join(
            format_signature_value(key, item, cls_name)
            for item in value)
    return format_signature_value(key, value, cls_name)


def snapshot_detail_sections(snapshot):
    """把一条快照的全部输入摊成 [(组名, [(字段名, 取值), ...]), ...]。

    分组、字段中文名和取值回译全部走对比页那一套（``_COMPARISON_ASPECTS``
    / ``_COMPARISON_FIELD_LABELS`` / ``option_param_labels`` /
    ``format_signature_value``），于是详情窗里读到的措辞与说明卡报差异时
    用的一定是同一个词。各写一套中文名的话，改了期权参数标签只同步一边，
    用户就会在两处看到同一项的两个名字。

    渲染成 ``—`` 的字段整条略过：签名为了让键集合恒定（见
    ``snapshot_strategy_signature``），会给与本条无关的项填 ``None``——
    每日收盘的快照带着 ``fixed_times=None``，Wind 行情的快照带着
    ``csv_path=None``。差异比对需要这些占位，人看的详情不需要，列出来只
    会让真正有值的那几行更难找。**合约参数组豁免这条规则**：详情窗是照着
    它重建合约的依据，少一项就重建不出来。「已实现序列」留空是合法取值
    （= 没有已过的日子）而不是缺失，渲染成 ``—`` 也要留在表里。每种期权
    定义了几项就列几项（香草 7、累计 15、亚式 12、气囊 11、雪球 18）。

    标签、取值回译和**行序**都按这条快照自己的期权大类取：``param_labels``
    把该大类的参数排在最前，于是详情里的参数顺序就是左侧表单里的定义顺序。
    """
    cls_name = signature_cls_name(
        [flatten_signature(
            tuple(getattr(snapshot, "contract_key", ()) or ()))])
    param_labels, _choices = option_param_labels(cls_name)
    order = list(_COMPARISON_FIELD_LABELS) + list(param_labels)
    sections = []
    for aspect, label, getter in _COMPARISON_ASPECTS:
        rows = []
        flat = flatten_signature(tuple(getter(snapshot)))
        # 行情组只列本次来源用得上的那几项；来源不认识时全列，宁可多几行
        # 也不要因为一个没见过的来源名把整组抹空。
        allowed = (
            _MARKET_DETAIL_FIELDS.get(flat.get(("source",)))
            if aspect == "market" else None)
        for path, value in flat.items():
            field = path[-1]
            if field in _SNAPSHOT_DETAIL_HIDDEN_FIELDS:
                continue
            if allowed is not None and field not in allowed:
                continue
            override = _DETAIL_VALUE_OVERRIDES.get(
                (aspect, field))
            shown = (
                override(snapshot, value) if override
                else format_detail_value(
                    field, value, cls_name))
            # 合约参数一项都不能少：详情窗是照着它重建合约的依据，
            # 「已实现序列」留空是合法取值（= 没有已过的日子），不是缺失。
            # 略过空值这条规则是为签名占位（fixed_times=None 之类）设的，
            # 不该把真实的期权参数一起吞掉。
            if shown == "—" and aspect != "contract":
                continue
            rows.append((
                order.index(field) if field in order else len(order),
                _COMPARISON_FIELD_LABELS.get(
                    field, param_labels.get(field, field)),
                shown,
            ))
        # 与差异串同一个排序依据：先按定义顺序，同 rank 再按文本。
        # flatten_signature 的键来自 frozenset 化的路径集合，不兜底按文
        # 本排的话，同一条快照两次打开可能排出不同的顺序。
        sections.append(
            (label, [(name, text) for _rank, name, text in sorted(rows)]))
    return sections


def history_row_form_state(row):
    """把优选排名行的 ``meta_*`` 还原成一份对冲策略 form_state。

    这几个字段正是「应用策略到左侧参数」读的那些，够拼出完整的策略签名。
    候选带宽一律以 σ 表达：整批候选共用左侧输入的波动率，排名比的就是 σ
    倍数本身（见 ``_collect_history_state``）。
    """
    fallback = row.get("meta_force_day_close_hedge")
    return {
        "strategy_name": str(
            row.get("meta_strategy_name")
            or row.get("strategy_type") or "").strip(),
        "fixed_times": str(row.get("meta_fixed_times", "") or ""),
        "interval_type": "sigma",
        "price_interval": history_selection.finite_value(
            row.get("meta_candidate_sigma")),
        "sigma_source": str(
            row.get("meta_sigma_source", "") or "implied"),
        "sigma_window": history_selection.safe_int(
            row.get("meta_sigma_window"), 20),
        "force_day_close_hedge": (
            bool(fallback)
            if isinstance(fallback, (bool, np.bool_)) else False),
    }


def history_row_detail_sections(row, history_state):
    """把优选排名的一行摊成可展示的分组。

    与结果池最大的不同是**参数天然分两层**：优选的全部候选共用同一套期权、
    行情与规模成本（那正是控制变量的做法），只有对冲策略逐条不同。所以这
    里拿运行时冻结的 state 造出前四组，再把这一行自己的 ``meta_*`` 拼成
    第五组的 form_state——五组的字段、中文名和取值回译于是与结果池详情完全
    同构，两个窗口不会对同一项给出两种说法。

    末尾另加一组「本行的取样口径」：它回答"这一行的数字是在多大样本上算
    出来的"，是优选特有的、结果池没有的一层。
    """
    history_state = dict(history_state or {})
    row = dict(row or {})
    market_key, contract_key, economics_key = (
        signature_keys_from_state(history_state))
    try:
        position = wind_resolve.normalize_position(
            history_state.get("position"))
    except ValueError:
        position = 1
    probe = SimpleNamespace(
        market_key=market_key,
        contract_key=contract_key,
        economics_key=economics_key,
        position=position,
        form_state=history_row_form_state(row),
        bar_size_requested=str(
            history_state.get("wind_bar_size_requested", "") or ""),
        intraday_steps=0,
    )
    sections = snapshot_detail_sections(probe)

    scope = []
    for key, label in (
            ("period", "周期"),
            ("strategy", "候选"),
    ):
        text = str(row.get(key, "") or "").strip()
        if text:
            scope.append((label, text))
    paired = history_selection.safe_int(
        row.get("paired_windows", row.get("rolling_windows")), 0)
    if paired:
        scope.append(("参与评分段数", str(paired)))
    baseline = history_selection.safe_int(
        row.get("baseline_windows"), 0)
    if baseline:
        scope.append(("基准段数", str(baseline)))
    sections.append(("本行的取样口径", scope))
    return sections


def comparison_aspect_diff(snapshots):
    """逐属性比对，返回 [(中文名, 是否相同, 差异字段列表), ...]。

    少于两条时没有"异同"可言，返回空表。
    """
    snapshots = list(snapshots)
    if len(snapshots) < 2:
        return []
    report = []
    for _key, label, getter in _COMPARISON_ASPECTS:
        values = [tuple(getter(snapshot)) for snapshot in snapshots]
        same = len(set(values)) == 1
        fields = [] if same else differing_field_names(values)
        report.append((label, same, fields))
    return report


def comparison_variable_summary(snapshots):
    """把属性异同拆成三层：标题、逐字段取值、其余一致。

    标题只说"变量是哪个属性"，长度恒定；具体取值交给 ``fields``，由展示
    层按序号排成网格。此前三层挤在一行粗体里，三条结果差两项就是七八十
    个字，还得靠 "A vs B vs C" 硬串——那串既看不出谁是谁，也没法截断。
    """
    report = comparison_aspect_diff(snapshots)
    if not report:
        return None
    changed = [(label, fields) for label, same, fields in report if not same]
    same_labels = [label for label, same, _f in report if same]

    def _rows(entries):
        """把各属性的字段摊平；字段名与属性名重合时不重复。"""
        rows = []
        for label, fields in entries:
            if not fields:
                rows.append((label, None))
                continue
            for name, values in fields:
                rows.append((name, values))
        return rows

    if not changed:
        return {
            "state": "identical",
            "headline": "所选结果的输入完全相同",
            "fields": [],
            "rest": ("行情、期权类型与参数、头寸方向、规模与成本、"
                     "对冲策略逐项一致；"
                     "若数字仍有差异，只可能来自未记录的输入。"),
        }
    if len(changed) == 1:
        return {
            "state": "single",
            "headline": f"本次对比的变量：{changed[0][0]}",
            "fields": _rows(changed),
            "rest": (f"其余一致：{'、'.join(same_labels)}"
                     if same_labels else ""),
        }
    return {
        "state": "multiple",
        "headline": (
            f"同时有 {len(changed)} 项不同："
            + "、".join(label for label, _fields in changed)),
        "fields": _rows(changed),
        "rest": ("差异无法归因到其中任何一项；"
                 + (f"一致的是：{'、'.join(same_labels)}"
                    if same_labels else "没有任何一项是一致的")),
    }


def saved_comparison_warnings(snapshots):
    """属性异同之外，还需要单独说明的看数限制。

    不报"行情路径不同"：模拟行情下价格序列由期权参数派生（改 σ 就会重
    新生成路径），Wind/CSV 下换区间也必然换数据——路径差异是上游输入
    变化的必然结果，单独说一句"行情数据本身不同"会和"其余一致：行情"
    直接打架。真正需要提醒的是**长度不同**：那时曲线按序号对齐才有误
    导性，累计类指标也不在同一个跨度上。
    """
    snapshots = list(snapshots)
    if len(snapshots) < 2:
        return []
    warnings = []
    lengths = {
        history_selection.safe_int(
            (snapshot.summary_row or {}).get("n_trade_days"), 0)
        for snapshot in snapshots
    }
    if len(lengths) > 1:
        warnings.append(
            "各条的交易日数不同：曲线按序号对齐，"
            "期末净损益与总成本不在同一个跨度上。")
    positions = {
        saved_snapshot_position(snapshot)
        for snapshot in snapshots
    }
    if len(positions) > 1:
        warnings.append(
            "同时包含买入与卖出：期末净损益与最大回撤的正负来自方向本身，"
            "横向看应比总成本、再触发与换手额。")
    return warnings


def copy_snapshot_gui_state(gui_state):
    """去掉不可序列化的构造器，仅保留本次运行的参数快照。"""
    return copy.deepcopy({
        key: value for key, value in gui_state.items() if key != "cfg"
    })
