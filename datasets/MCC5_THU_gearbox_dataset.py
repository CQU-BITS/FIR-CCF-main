"""
MCC5-THU Gearbox dataset
Available at: https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets
Reference   : S. Chen, Z. Liu, X. He, D. Zou, D. Zhou, Multi-mode fault diagnosis datasets of gearbox under variable
             working conditions, Data in Brief, 54 (2024) 110453.
"""


import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.sequence_aug import *
from tqdm import tqdm
from torch.utils.data import Dataset

Fs = 12800  # sampling frequency
sample_len = 2048  # the length of samples
lab = [0, 1, 2, 3, 4, 5, 6, 7]  # labels
data_load_path = 'V:/MCC5-THU齿轮箱数据集/原始数据集/'

# 1000rpm, 0~20Nm
WC1000 = ['health_torque_circulation_1000rpm_20Nm.csv',
          'teeth_crack_M_torque_circulation_1000rpm_20Nm.csv',
          'gear_wear_M_torque_circulation_1000rpm_20Nm.csv',
          'teeth_break_M_torque_circulation_1000rpm_20Nm.csv',
          'gear_pitting_M_torque_circulation_1000rpm_20Nm.csv',
          'miss_teeth_torque_circulation_1000rpm_20Nm.csv',
          'teeth_break_and_bearing_inner_M_torque_circulation_1000rpm_20Nm.csv',
          'teeth_break_and_bearing_outer_M_torque_circulation_1000rpm_20Nm.csv']
# 2000rpm, 0~20Nm
WC2000 = ['health_torque_circulation_2000rpm_20Nm.csv',
          'teeth_crack_M_torque_circulation_2000rpm_20Nm.csv',
          'gear_wear_M_torque_circulation_2000rpm_20Nm.csv',
          'teeth_break_M_torque_circulation_2000rpm_20Nm.csv',
          'gear_pitting_M_torque_circulation_2000rpm_20Nm.csv',
          'miss_teeth_torque_circulation_2000rpm_20Nm.csv',
          'teeth_break_and_bearing_inner_M_torque_circulation_2000rpm_20Nm.csv',
          'teeth_break_and_bearing_outer_M_torque_circulation_2000rpm_20Nm.csv']
# 3000rpm, 0~20Nm
WC3000 = ['health_torque_circulation_3000rpm_20Nm.csv',
          'teeth_crack_M_torque_circulation_3000rpm_20Nm.csv',
          'gear_wear_M_torque_circulation_3000rpm_20Nm.csv',
          'teeth_break_M_torque_circulation_3000rpm_20Nm.csv',
          'gear_pitting_M_torque_circulation_3000rpm_20Nm.csv',
          'miss_teeth_torque_circulation_3000rpm_20Nm.csv',
          'teeth_break_and_bearing_inner_M_torque_circulation_3000rpm_20Nm.csv',
          'teeth_break_and_bearing_outer_M_torque_circulation_3000rpm_20Nm.csv']

def get_data(speed, snr):
    """
    This function is used to generate the training set and test set.
    :param speed: The rotating speed in rpm
    :return:
    """

    data, labels = [], []
    if speed == 1000:  # 1000rpm, 0~20Nm
        WC = WC1000
    elif speed == 2000:  # 2000rpm, 0~20Nm
        WC = WC2000
    elif speed == 3000:  # 3000rpm, 0~20Nm
        WC = WC3000
    else:
        raise ValueError('This working condition is not included!')

    for ii in range(len(WC)):
        filename = WC[ii]
        data_root = os.path.join(data_load_path, filename)
        # select signals form channel “gearbox_vibration_z”
        signal = pd.read_csv(data_root, header=0, usecols=[7]).values.squeeze()
        # select vibration signals under steady torque at 11~19 and 41~49 seconds
        idx = np.arange(11*Fs, 19*Fs).tolist() + np.arange(41*Fs, 49*Fs).tolist()
        signal = signal[idx]

        if snr != None:
            signal = Add_noise(snr)(signal)

        start, end = 0, sample_len
        while end <= signal.shape[0]:
            data.append(signal[start: end])
            labels.append(lab[ii])
            start += sample_len

            end += sample_len
    data = np.array(data)
    data = np.expand_dims(data, axis=1)  # data >> [num, 1, sample_len]
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = self.data[item]
        label = self.labels[item]
        return seq, label


class MCC5_THU_gearbox_dataset(object):
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

        train_X5 = np.concatenate((train_X, train_X[np.where(train_Y == 5)[0]].repeat(len(lab) - 2, 0)), axis=0)
        train_Y5 = np.ones([train_X5.shape[0]])
        train_Y5[np.where(train_Y != 5)[0]] = 0

        train_X6 = np.concatenate((train_X, train_X[np.where(train_Y == 6)[0]].repeat(len(lab) - 2, 0)), axis=0)
        train_Y6 = np.ones([train_X6.shape[0]])
        train_Y6[np.where(train_Y != 6)[0]] = 0

        train_X7 = np.concatenate((train_X, train_X[np.where(train_Y == 7)[0]].repeat(len(lab) - 2, 0)), axis=0)
        train_Y7 = np.ones([train_X7.shape[0]])
        train_Y7[np.where(train_Y != 7)[0]] = 0

        real_train_set = {"data": train_X, "label": train_Y}
        # real_val_set = {"data": val_X, "label": val_Y}
        real_test_set = {"data": test_X, "label": test_Y}

        fake__train_set0 = {"data": train_X0, "label": train_Y0}
        fake__train_set1 = {"data": train_X1, "label": train_Y1}
        fake__train_set2 = {"data": train_X2, "label": train_Y2}
        fake__train_set3 = {"data": train_X3, "label": train_Y3}
        fake__train_set4 = {"data": train_X4, "label": train_Y4}
        fake__train_set5 = {"data": train_X5, "label": train_Y5}
        fake__train_set6 = {"data": train_X6, "label": train_Y6}
        fake__train_set7 = {"data": train_X7, "label": train_Y7}

        real_train_set = dataset(real_train_set, transform=data_transforms(self.normlizetype))
        # real_val_set = dataset(real_val_set, transform=data_transforms(self.normlizetype))
        real_test_set = dataset(real_test_set, transform=data_transforms(self.normlizetype))
        fake__train_set0 = dataset(fake__train_set0, transform=data_transforms(self.normlizetype))
        fake__train_set1 = dataset(fake__train_set1, transform=data_transforms(self.normlizetype))
        fake__train_set2 = dataset(fake__train_set2, transform=data_transforms(self.normlizetype))
        fake__train_set3 = dataset(fake__train_set3, transform=data_transforms(self.normlizetype))
        fake__train_set4 = dataset(fake__train_set4, transform=data_transforms(self.normlizetype))
        fake__train_set5 = dataset(fake__train_set5, transform=data_transforms(self.normlizetype))
        fake__train_set6 = dataset(fake__train_set6, transform=data_transforms(self.normlizetype))
        fake__train_set7 = dataset(fake__train_set7, transform=data_transforms(self.normlizetype))
        # return real_train_set, real_test_set
        return fake__train_set0, fake__train_set1, fake__train_set2, fake__train_set3, fake__train_set4, \
               fake__train_set5, fake__train_set6, fake__train_set7, real_train_set, real_test_set



if __name__ == '__main__':
    fake__train_set0, fake__train_set1, fake__train_set2, fake__train_set3, fake__train_set4, \
    fake__train_set5, fake__train_set6, fake__train_set7, real_train_set, real_test_set \
        = MCC5_THU_gearbox_dataset(speed=1000, snr=None, normlizetype='mean~std', shot=1).data_preprare()
    data = real_train_set.data
    print(np.mean(data[1]), np.std(data[1]))
    print(np.mean(data), np.std(data))
    pass

