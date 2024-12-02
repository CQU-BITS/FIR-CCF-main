# Authors   : Rui Liu, Xiaoxi Ding, Shenglan Liu, Hebin Zheng, Yuanyaun Xu, Yimin Shao
# URL       : https://www.sciencedirect.com/science/article/pii/S0951832024006811
# Reference : R. Liu, X. Ding, S. Liu, H. Zheng, Y. Xu, Y. Shao, Knowledge-informed FIR-based cross-category filtering framework for
#             interpretable machinery fault diagnosis under small samples, Reliab. Eng. Syst. Saf., 254 (2025) 110610.
# DOI       : https://doi.org/10.1016/j.ress.2024.110610
# Date      : 2024/12/02
# Version   : V0.1.0
# Copyright by CQU-BITS


import torch
from torch import nn

device = torch.device('cuda:1')


class SubNet(nn.Module):
    def __init__(self, ):
        super(SubNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.GAP(out)
        out = self.flatten(out)
        return out


class BCBLearner(nn.Module):
    def __init__(self, num_CKs=12, len_CKs=3, num_classes=2):
        super(BCBLearner, self).__init__()
        self.num_CKs = num_CKs
        self.len_CKs = len_CKs
        self.num_classes = num_classes
        # 将FIR滤波和替换为常规的卷积核 Conv(12 @ 1×3）
        self.Conv = nn.Conv1d(1, self.num_CKs, kernel_size=self.len_CKs, stride=1, padding=1)
        self.BN = nn.BatchNorm1d(self.num_CKs)
        # 每个模态提特征，输出长 16 的特征向量
        for ii in range(16):
            exec('self.subNet' + str(ii + 1) + '= SubNet()')
        self.two_class_classifier = nn.Linear(16 * self.num_CKs, num_classes)

    def forward(self, inputs):
        modes = self.Conv(inputs)
        out = self.BN(modes)
        for ii in range(self.num_CKs):
            exec('V' + str(ii+1) + '= self.subNet' + str(ii+1) + '(out[:, ii:ii+1, :])')
        F = torch.tensor([]).to(inputs.device)
        for ii in range(self.num_CKs):
            F = torch.cat((F, eval('V' + str(ii+1))), dim=1)

        out = self.two_class_classifier(F)  # [num, 16 * self.num]

        return modes, F, out


class Multi_class_classifier(nn.Module):
    def __init__(self, num_CKs, num_classes):
        super(Multi_class_classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(16 * num_CKs * num_classes, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes))

    def forward(self, x):
        return self.fc(x)


class Conv_CCF(nn.Module):
    def __init__(self, num_CKs, num_classes, *args, **kwargs):
        super(Conv_CCF, self).__init__()
        self.num_classes = num_classes
        # step 1: 多任务学习，每个学习任务基于二值化的伪标签完成跨类滤波与分类

        for ii in range(self.num_classes):
            exec('self.BCBLearner' + str(ii + 1) + '= kwargs[\"BCBLearner' + str(ii + 1) + '\"]')

        # step 2: （参数固定）提取所有任务中的 feature_extractor，然后基于真实标签训练Global_classifier
        self.classifier = Multi_class_classifier(num_CKs=num_CKs, num_classes=num_classes)

    def forward(self, inputs):
        for ii in range(self.num_classes):
            exec('m'+ str(ii+1) + ', F' + str(ii+1) + ', y' + str(ii+1) +  '= self.BCBLearner' + str(ii+1) + '(inputs)')

        V = torch.tensor([]).to(inputs.device)
        for ii in range(self.num_classes):
            V = torch.cat((V, eval('F' + str(ii+1))), dim=1)  # [num, 16 * self.num * num_classes]
        out = self.classifier(V)
        return out


if __name__ == '__main__':
    temp = torch.randn([32, 1, 2048]).to(device)
    BCBLearner1 = BCBLearner(num_CKs=12, len_CKs=3, num_classes=2).to(device)
    BCBLearner2 = BCBLearner(num_CKs=12, len_CKs=3, num_classes=2).to(device)
    BCBLearner3 = BCBLearner(num_CKs=12, len_CKs=3, num_classes=2).to(device)
    BCBLearner4 = BCBLearner(num_CKs=12, len_CKs=3, num_classes=2).to(device)
    BCBLearner5 = BCBLearner(num_CKs=12, len_CKs=3, num_classes=2).to(device)
    BCBLearner6 = BCBLearner(num_CKs=12, len_CKs=3, num_classes=2).to(device)
    BCBLearner7 = BCBLearner(num_CKs=12, len_CKs=3, num_classes=2).to(device)
    BCBLearner8 = BCBLearner(num_CKs=12, len_CKs=3, num_classes=2).to(device)
    model = Conv_CCF(num_CKs=12, num_classes=8,
                     BCBLearner1=BCBLearner1, BCBLearner2=BCBLearner2,
                     BCBLearner3=BCBLearner3, BCBLearner4=BCBLearner4,
                     BCBLearner5=BCBLearner5, BCBLearner6=BCBLearner6,
                     BCBLearner7=BCBLearner7, BCBLearner8=BCBLearner8).to(device)
    out = model(temp)
    print(out.shape)

    for name, param in model.named_parameters():
        print(name)

