# encoding: utf-8
"""
@project = AI_Studio
@file = 1
@author = GRIT
@create_time = 2021/7/31 17:19
"""
import numpy as np

# 从左到右分别是：[是否受伤, 是否胜利, 是否难过]
InputHiddent = np.array([[0.2, 0.01, 0.05],  # 隐藏层第一个单元
                         [0.01, 0.025, 0.04],  # 隐藏层第二个单元
                         [0.13, 0.03, 0.001]])  # 隐藏层第三个单元

# 从左到右分别是：隐藏层第一个单元到第三个单元的权重
HiddentPrediction = np.array([[0.02, 0.1, 0.5],  # 胜负记录
                              [0.01, 0.25, 0.04],  # 粉丝数量
                              [0.013, 0.3, 0.01]])  # 球员数量

weights = [InputHiddent, HiddentPrediction]  # 存放权重值，[输入到隐藏层的权重， 隐藏层到输出的权重]
inputs = np.array([0.65, 12.0, 8.0])  # 分别是该球队的胜负记录、粉丝数量、球员数量


def NeuralNetwork(inputs, weights):
    hid = inputs.dot(weights[0])  # 计算隐藏层的输出
    pred = hid.dot(weights[1])  # 将上一层（隐藏层）的输出带入下一层的输入
    return pred


pred = NeuralNetwork(inputs, weights)
print(pred)
print("受伤比例预测：{}".format(pred[0]))
print("比赛胜负预测：{}".format(pred[1]))
print("悲伤程度预测：{}".format(pred[2]))
