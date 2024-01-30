# _*_ coding: utf-8 _*_
"""
Created on 11月 30 17:09 2023 

@author: Grefer
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from WindPy import w


class VolatilityCone(object):
    def __init__(self,code):
        self.code = code

    def get_data(self):
        code_s = self.code.split('.')
        optionset = w.wset("optionfuturescontractbasicinf", "exchange=%s;productcode=%s;contract=all" % (code_s[1],code_s[0]))












