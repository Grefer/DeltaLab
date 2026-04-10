# _*_ coding: utf-8 _*_
"""
Created on 11月 30 17:09 2023 

@author: Grefer
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from WindPy import w

w.start() # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒 

w.isconnected() # 判断WindPy是否已经登录成功

data = w.wsd("000001.SH", "close", "2023-01-01", "2023-12-31", "") # 获取上证指数2023年全年的收盘价数据
df = pd.DataFrame(data.Data, index=data.Fields, columns=data.Times).T #










