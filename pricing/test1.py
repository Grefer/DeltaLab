# _*_ coding: utf-8 _*_
"""
Created on 6月 13 14:13 2024 

@author: Grefer
"""
import numpy as np

def enhanced_asian_option_pricing(S0, K, T, r, sigma, touch_price, enhance_price, N, M):
    """
    S0 : 初始股票价格
    K : 行权价格
    T : 到期时间
    r : 无风险利率
    sigma : 波动率
    touch_price : 触碰价格
    enhance_price : 增强价格
    N : 时间步长
    M : 模拟路径数量
    """
    dt = T/N
    stock_price = np.zeros((N+1, M))
    stock_price[0] = S0
    touch_flag = np.zeros(M, dtype=bool)

    # 模拟股票价格路径
    for t in range(1, N+1):
        brownian = np.random.standard_normal(M)
        stock_price[t] = stock_price[t-1]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*brownian)

        # 检查是否触及触碰价格
        touch_flag = np.logical_or(touch_flag, stock_price[t] >= touch_price)

        # 根据是否触及触碰价格，决定结算价格
        stock_price[t] = np.where(touch_flag, np.maximum(stock_price[t], touch_price), np.maximum(stock_price[t], enhance_price))

    # 计算亚式期权的平均价格
    average_price = np.mean(stock_price, axis=0)

    # 计算期权的内在价值
    option_value = np.maximum(average_price-K, 0)

    # 使用无风险利率折现
    return np.exp(-r*T)*np.mean(option_value)

# 测试参数
if __name__ == '__main__':
    S0 = 100
    K = 100
    T = 22/243.
    r = 0.00
    sigma = 0.12
    touch_price = 120
    enhance_price = 100
    N = 22
    M = 100000
    price = enhanced_asian_option_pricing(S0, K, T, r, sigma, touch_price, enhance_price, N, M)
print(price)