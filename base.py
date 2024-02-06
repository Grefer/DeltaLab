# _*_ coding: utf-8 _*_
"""
Created on 1月 30 11:23 2024 

@author: Grefer
"""
import sys
sys.path.append('/Users/Grefer/Documents/GitHub/Quant/pricing')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Option_AS


def get_trades():
    filename = r'C:\Users\Grefer\OneDrive\Work4life\保险+期货对冲表修订版.xlsm'
    data = pd.read_excel(filename, sheet_name='项目台账',skiprows=1, engine='openpyxl')

    return data

def get_option(data):
    attribute_list = ['optiontype','s0','sr','K','E','T','N','sigma','cp','minPay','maxPay']
    option = []
    for index in data.index.values:

        option.append(Option_AS(data['产品类型'],data['最新价'],data['']))








if __name__ == '__main__':
    get_trades()