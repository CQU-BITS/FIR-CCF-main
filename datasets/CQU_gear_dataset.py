"""
CQU (Chongqing University) gearbox dataset form a self-made two-stage Gearbox
Available at:
# Reference : R. Liu, X. Ding, S. Liu, H. Zheng, Y. Xu, Y. Shao, Knowledge-informed FIR-based cross-category filtering framework for
#             interpretable machinery fault diagnosis under small samples, Reliab. Eng. Syst. Saf., 254 (2025) 110610.
"""


import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.sequence_aug import *
from tqdm import tqdm
from torch.utils.data import Dataset


Fs = 10240  # 降采样数据集的采样频率
sample_len = 2048  # the length of samples
lab = [0, 1, 2, 3, 4]  # The filename of vibration signals data was labeled to 0-4
data_load_path = 'R:/★My Researches/★公开数据集/CQU gearbox dataset/ConstantSpeed_downsampled_5120/'


def get_data(speed, snr):
    """
    This function is used to generate the training set and test set.
    :param speed: The rotating speed in rpm
    :return:
    """

    data, labels = [], []
    for ii in range(len(lab)):
        filename = str(lab[ii]+1) + '_' + str(speed) + '_0.02.mat'
        data_root = os.path.join(data_load_path, filename)
        signal = loadmat(data_root)['Signal'][0][0]['y_values'][0][0]['values'][:, [1]]  # 通道2（传感器v2)的数据被使用
        if snr != None:
            signal = Add_noise(snr)(signal)

        start, end = 0, sample_len
        while end <= signal.shape[0]:
            data.append(signal[start: end])
            labels.append(lab[ii])
            start += sample_len
            end += sample_len
    data = np.array(data)
    data = np.transpose(data, (0, 2, 1))  # data >> [num, 1, sample_len]
    labels = np.array(labels)  # label >> [num]
    print(data.shape, labels.shape)
    return data, labels


def data_transforms(normlize_type=None):
    transforms = Compose([
        Normalize(normlize_type),
        Retype()])
    return transforms


class dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.transforms = transform

        self.data = list(self.transforms(dataset['data']))
        self.labels = dataset['label'].tolist()
        # self.labels = dataset['label'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = self.data[item]
        label = self.labels[item]
        return seq, label


class CQU_gear_dataset(object):
    in_channels = 1
    num_classes = len(lab)

    def __init__(self, speed, snr, normlizetype, shot):
        self.speed = speed
        self.snr = snr
        self.normlizetype = normlizetype
        self.shot = shot

    def data_preprare(self):
        data, labels = get_data(speed=self.speed, snr=self.snr)
        # 存在验证集时
        train_X, test_X, train_Y, test_Y = train_test_split(data, labels, train_size=len(lab) * self.shot, shuffle=True, stratify=labels)
        # val_X, val_Y = test_X[0:400, :, :], test_Y[0: 400]
        # val_X, val_Y = test_X, test_Y

        # 生产带伪标签的子任务训练集，防止类不平衡的影响，为标签为 1 的样本复制 K-2 次
        train_X0 = np.concatenate((train_X, train_X[np.where(train_Y == 0)[0]].repeat(len(lab) - 2, 0)), axis=0)
        train_Y0 = np.ones([train_X0.shape[0]])
        train_Y0[np.where(train_Y != 0)[0]] = 0

        train_X1 = np.concatenate((train_X, train_X[np.where(train_Y == 1)[0]].repeat(len(lab) - 2, 0)), axis=0)
        train_Y1 = np.ones([train_X1.shape[0]])
        train_Y1[np.where(train_Y != 1)[0]] = 0

        train_X2 = np.concatenate((train_X, train_X[np.where(train_Y == 2)[0]].repeat(len(lab) - 2, 0)), axis=0)
        train_Y2 = np.ones([train_X2.shape[0]])
        train_Y2[np.where(train_Y != 2)[0]] = 0

        train_X3 = np.concatenate((train_X, train_X[np.where(train_Y == 3)[0]].repeat(len(lab) - 2, 0)), axis=0)
        train_Y3 = np.ones([train_X3.shape[0]])
        train_Y3[np.where(train_Y != 3)[0]] = 0

        train_X4 = np.concatenate((train_X, train_X[np.where(train_Y == 4)[0]].repeat(len(lab) - 2, 0)), axis=0)
        train_Y4 = np.ones([train_X4.shape[0]])
        train_Y4[np.where(train_Y != 4)[0]] = 0


        real_train_set = {"data": train_X, "label": train_Y}
        real_test_set = {"data": test_X, "label": test_Y}

        fake__train_set0 = {"data": train_X0, "label": train_Y0}
        fake__train_set1 = {"data": train_X1, "label": train_Y1}
        fake__train_set2 = {"data": train_X2, "label": train_Y2}
        fake__train_set3 = {"data": train_X3, "label": train_Y3}
        fake__train_set4 = {"data": train_X4, "label": train_Y4}

        real_train_set = dataset(real_train_set, transform=data_transforms(self.normlizetype))
        real_test_set = dataset(real_test_set, transform=data_transforms(self.normlizetype))
        fake__train_set0 = dataset(fake__train_set0, transform=data_transforms(self.normlizetype))
        fake__train_set1 = dataset(fake__train_set1, transform=data_transforms(self.normlizetype))
        fake__train_set2 = dataset(fake__train_set2, transform=data_transforms(self.normlizetype))
        fake__train_set3 = dataset(fake__train_set3, transform=data_transforms(self.normlizetype))
        fake__train_set4 = dataset(fake__train_set4, transform=data_transforms(self.normlizetype))

        return fake__train_set0, fake__train_set1, fake__train_set2, fake__train_set3, fake__train_set4, \
               real_train_set, real_test_set


if __name__ == '__main__':
    fake__train_set0, fake__train_set1, fake__train_set2, fake__train_set3, fake__train_set4, \
    real_train_set, real_test_set = CQU_gear_dataset(speed=400, snr=None, normlizetype=None, shot=1).data_preprare()
    data = fake__train_set0.data
    print(np.mean(data[1]), np.std(data[1]))
    print(np.mean(data), np.std(data))
    pass

