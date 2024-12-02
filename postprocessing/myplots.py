# Authors   : Rui Liu, Xiaoxi Ding, Shenglan Liu, Hebin Zheng, Yuanyaun Xu, Yimin Shao
# URL       : https://www.sciencedirect.com/science/article/pii/S0951832024006811
# Reference : R. Liu, X. Ding, S. Liu, H. Zheng, Y. Xu, Y. Shao, Knowledge-informed FIR-based cross-category filtering framework for
#             interpretable machinery fault diagnosis under small samples, Reliab. Eng. Syst. Saf., 254 (2025) 110610.
# DOI       : https://doi.org/10.1016/j.ress.2024.110610
# Date      : 2024/12/02
# Version   : V0.1.0
# Copyright by CQU-BITS


import pandas as pd
import numpy as np
import os
from pylab import *
from sklearn.manifold import TSNE
import itertools
from sklearn.metrics import confusion_matrix


plt.rcParams['figure.dpi'] = 600  # plt.show显示分辨率
plt.rcParams['axes.unicode_minus'] = False
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 16}
plt.rc('font', **font)


# 绘图颜色和标记符号
color = ['black', 'red', 'blue', 'green', 'cyan', 'magenta', 'darkkhaki', 'gray', 'blueviolet', 'olive', 'brown',
         'plum', 'maroon', 'yellow', 'salmon']
# marker = ['o', 's', '^', 'p', 'X', '*', '8', 'D', '+', '<', '>', 'x', 'P', 'd', 'H']
marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']

def plotLossAcc(trainLoss, testLoss, trainAcc, testAcc, save_path, fig_name):

    epochs = np.arange(len(trainLoss))
    fig, subs = plt.subplots(2, figsize=(4, 7))  # 返回画布和子图

    subs[0].plot(epochs, trainLoss, 'g-', label='Training', lw=2)
    subs[0].plot(epochs, testLoss, 'r-', label='testing', lw=2)
    subs[0].set_xlabel('Epoch')
    subs[0].set_ylabel('Loss')
    subs[0].legend()

    subs[1].plot(epochs, trainAcc, 'g-', label='Training', lw=2)
    subs[1].plot(epochs, testAcc, 'r-', label='testing', lw=2)
    subs[1].set_xlabel('Epoch')
    subs[1].set_ylabel('Acc. (%)')
    subs[1].legend()
    plt.tight_layout()
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    plt.savefig(os.path.join(save_path, fig_name), dpi=600, bbox_inches='tight')
    plt.show()



def plotTSNECluster(dataset, typeNum, typeLabel, save_path, fig_name):

    # 分离数据和标签，将数据用 PCA 降维至2维
    data, labels = dataset[:, 0:-1], dataset[:, -1]
    data_tSNE = TSNE(n_components=2, init='pca', learning_rate=200, method='exact', random_state=0).fit_transform(data)

    # 对每一类故障数据，索引其 lable对应降维后的特征并存为变量 type
    for ii in range(typeNum):
        idx = np.where(np.array(labels) == ii)
        globals()['type'+str(ii)] = np.array(data_tSNE)[idx]
    plt.figure(figsize=(4.5, 3))
    for jj in range(typeNum):
        plt.scatter(eval("type"+str(jj))[:, 0], eval("type"+str(jj))[:, 1], s=10, c=color[jj], marker=marker[jj])
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(labels=typeLabel, fontsize=13, loc='best')
    # plt.legend(labels=typeLabel, fontsize=13, ncol=typeNum, handletextpad=0, borderaxespad=0.1, columnspacing=0.1,
    #            loc='upper center', bbox_to_anchor=(0, 1.350), edgecolor='black')
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    plt.savefig(os.path.join(save_path, fig_name), dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plotConfusionMatrix(pred, target, class_names, save_path, fig_name):

    cmtx = confusion_matrix(target, pred)
    num_classes = len(class_names)

    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=(4, 4))
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        # 显示数值
        plt.text(j, i, format(cmtx[i, j], "") if cmtx[i, j] != 0 else "0",
                 horizontalalignment="center",
                 verticalalignment='center', color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    os.makedirs(save_path) if not os.path.exists(save_path) else None
    plt.savefig(os.path.join(save_path, fig_name), dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

