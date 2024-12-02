"""
SEU (Southeast University) bearing dataset
Available at: https://github.com/cathysiyu/Mechanical-datasets
Reference   : S.Y. Shao, S. McAleer, R.Q. Yan, P. Baldi, Highly Accurate Machine Fault Diagnosis Using Deep
              Transfer Learning, IEEE Trans. Ind. Inform., 15 (2019) 2446-2455.
"""


import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.sequence_aug import *
from tqdm import tqdm
from torch.utils.data import Dataset


sample_len = 2048  # the length of samples
lab = [0, 1, 2, 3, 4]  # The filename of vibration signals data was labeled to 0-4
home = 'R:/★My Researches/★公开数据集/SEU gearbox dataset/gearbox/bearingset'

WC1 = ['health_20_0.csv', 'outer_20_0.csv', 'inner_20_0.csv', 'ball_20_0.csv',  'comb_20_0.csv']
WC2 = ['health_30_2.csv', 'outer_30_2.csv', 'inner_30_2.csv', 'ball_30_2.csv',  'comb_30_2.csv']



def get_data(speed, snr):
    """
    This function is used to generate the training set and test set.
    :param speed: The rotating speed in rpm
    :return:
    """

    data, labels = [], []
    if speed == 20:  # 20Hz-0V
        WC = WC1
    elif speed == 30:  # 30Hz-2V
        WC = WC2
    else:
        raise NameError('This working condition is not included!')

    for ii in range(len(WC)):
        filename = WC[ii]
        data_root = os.path.join(home, filename)
        if filename == 'ball_20_0.csv':
            signal = pd.read_csv(data_root, header=15, sep=',').values
        else:
            signal = pd.read_csv(data_root, header=15, sep='\t').values
        signal = signal[24499:, 3]   # the fourth row of vibration signals were used here

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


class SEU_bearing_dataset(object):
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
        # real_val_set = {"data": val_X, "label": val_Y}
        real_test_set = {"data": test_X, "label": test_Y}

        fake__train_set0 = {"data": train_X0, "label": train_Y0}
        fake__train_set1 = {"data": train_X1, "label": train_Y1}
        fake__train_set2 = {"data": train_X2, "label": train_Y2}
        fake__train_set3 = {"data": train_X3, "label": train_Y3}
        fake__train_set4 = {"data": train_X4, "label": train_Y4}

        real_train_set = dataset(real_train_set, transform=data_transforms(self.normlizetype))
        # real_val_set = dataset(real_val_set, transform=data_transforms(self.normlizetype))
        real_test_set = dataset(real_test_set, transform=data_transforms(self.normlizetype))
        fake__train_set0 = dataset(fake__train_set0, transform=data_transforms(self.normlizetype))
        fake__train_set1 = dataset(fake__train_set1, transform=data_transforms(self.normlizetype))
        fake__train_set2 = dataset(fake__train_set2, transform=data_transforms(self.normlizetype))
        fake__train_set3 = dataset(fake__train_set3, transform=data_transforms(self.normlizetype))
        fake__train_set4 = dataset(fake__train_set4, transform=data_transforms(self.normlizetype))
        # return real_train_set, real_test_set
        return fake__train_set0, fake__train_set1, fake__train_set2, fake__train_set3, fake__train_set4, \
               real_train_set, real_test_set



if __name__ == '__main__':
    fake__train_set0, fake__train_set1, fake__train_set2, fake__train_set3, fake__train_set4, \
    real_train_set, real_test_set = SEU_bearing_dataset(speed=20, snr=None, normlizetype='mean~std',
                                                                      shot=1).data_preprare()
    data = real_train_set.data
    print(np.mean(data[1]), np.std(data[1]))
    print(np.mean(data), np.std(data))
    pass

