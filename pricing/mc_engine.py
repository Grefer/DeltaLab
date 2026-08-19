# _*_ coding: utf-8 _*_
"""
蒙特卡洛路径生成模块
"""

import numpy as np


def McGbmQ(
        s0: float,
        r: float,
        sigma: float,
        T: float,
        nPath: int,
        nStep: int,
        seed=20,
        draw_steps=None
):
    """
    GBM 蒙特卡洛路径生成（含对偶变量方差缩减）

    Parameters
    ----------
    s0 : 初始价格
    r : 漂移率 (通常为 r - q)
    sigma : 波动率
    T : 到期时间（年化）
    nPath : 模拟路径数（必须为偶数）
    nStep : 时间步数
    seed : int | None
        随机数种子：传 int 时走确定性序列（CRN / 可复现）；
        传 None 时用操作系统熵，彻底独立采样。默认 20（历史行为）。
    draw_steps : int | None
        **抽样时的步数**，默认与 nStep 相同。给得比 nStep 大时，先抽
        ``(nPath/2, draw_steps)``，再只用前 nStep 列。

        这是给 theta 用的：theta 的 bump 把剩余期限减一天，nStep 从 T 变成
        T−1，于是 ``standard_normal`` 换了形状、抽出**完全不同**的一批随机
        数——theta 就成了两次相互独立的 MC 估计之差，噪声被放大一个数量级。
        让 bump 也按 T 抽、再切前 T−1 列，两边就共享同一批随机数，CRN 成立。
        （行优先填充下，``(n, T)`` 的前 T−1 列 ≠ ``(n, T−1)``，所以必须让
        两边都按 T 抽，不能指望切片自动对上。）

    Returns
    -------
    s : shape (nPath, nStep) 的价格路径矩阵
    """
    if nPath <= 0 or nPath % 2 != 0:
        raise ValueError(f"nPath must be a positive even integer, got {nPath}")
    if nStep <= 0:
        raise ValueError(f"nStep must be positive, got {nStep}")
    draw = nStep if draw_steps is None else int(draw_steps)
    if draw < nStep:
        raise ValueError(
            f"draw_steps must be >= nStep, got {draw} < {nStep}")

    # seed=None 由 np.random.default_rng 自动走 OS 熵，保证每次采样独立。
    rng = np.random.default_rng(seed)

    # 全程在**一块**输出缓冲上原地推进。此前每一步都新建数组：
    # W1、np.r_ 拼出的 W、两个算术临时、cumsum、exp，峰值是结果矩阵的
    # 4.5 倍（nPath=1e5 / nStep=243 时 834 MB → 现在 185 MB）。
    #
    # **这是省内存，不是提速。** 完整优选链路上背靠背 A/B 七轮，
    # 中位数之比 1.01x——落在噪声里。原因是原地版仍要对同一块数组做五趟
    # 读写（乘、加、cumsum、exp、乘），带宽流量几乎没变，省掉的是分配。
    # 收益在峰值占用：雪球这类 payoff 很轻的结构，单次定价峰值从约 167 MB
    # 降到 42 MB，分段并行的内存预算因此宽裕得多。
    #
    # 每一步都与原写法逐位等价，随机流也没变：
    #   * standard_normal(out=) 只决定写到哪里，抽样序列不受影响；
    #   * 先乘 vol 再加 drift，与原来的 drift + vol*W 是同一组 IEEE754 运算
    #     （加法可交换，逐位相同）；
    #   * 最后的 s *= s0 与原来的 s0 * exp(...) 同理。
    # tests/test_pricing_memo.py 有一条把两种写法钉在一起的回归测试。
    half = nPath // 2
    buf = np.empty((nPath, draw), dtype=np.float64)
    rng.standard_normal((half, draw), out=buf[:half])
    np.negative(buf[:half], out=buf[half:])   # 对偶变量方差缩减
    # draw == nStep 时这就是整块缓冲；更大时是它的前 nStep 列视图，
    # 后面的原地运算照常作用在视图上。
    s = buf[:, :nStep]

    h = T / nStep
    s *= sigma * np.sqrt(h)
    s += (r - 0.5 * sigma ** 2) * h
    np.cumsum(s, axis=1, out=s)
    np.exp(s, out=s)
    s *= s0
    return s
