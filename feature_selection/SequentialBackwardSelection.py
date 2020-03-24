"""
SBS算法原理：从原始特征集中移除特征，直到新的特征集数目达到事先确定的值。而在移除特征时，我们需要定义评价函数J。
一个特征的重要性就用特征移除后的评价函数值表示。我们每一次都把那些评价函数值最大的特征移除，也就是那些对评价函数影响最小的特征去掉。
所以，SBS算法有以下4个步骤：
1、初始化k=d，其中d是原始特征维度
2、确定那个评价函数最大的特征
3、从Xk中移除特征x~，k=k-1
4、如果k等于事先确定的阈值则终止；否则回到步骤2
"""

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():

    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
