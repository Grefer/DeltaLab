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
        seed: int = 20
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
    seed : 随机数种子

    Returns
    -------
    s : shape (nPath, nStep) 的价格路径矩阵
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((nPath // 2, nStep))
    W = np.r_[W1, -W1]  # 对偶变量方差缩减
    h = T / nStep
    dlogS = (r - 0.5 * sigma ** 2) * h + sigma * np.sqrt(h) * W
    s = s0 * np.exp(np.cumsum(dlogS, 1))
    return s
