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
from math import pi
import torch.nn.functional as F


def Hanning(len_FIRs, wc, bw):
    L = len_FIRs
    deltaW = 6.2 * torch.pi / L
    n = torch.linspace(0, L-1, L).unsqueeze(dim=0).to(wc.device)
    wn = 0.5 * (1 - torch.cos(2 * torch.pi * n / (L - 1)))
    lamdan = (2*n-L+1)/4 - 0.0001
    hdn = (torch.sin(lamdan * (2*wc+bw+deltaW)) - torch.sin(lamdan * (2*wc-bw-deltaW))) / (2*lamdan*pi)
    hn = hdn * wn
    return hn


class FIRFiltering(nn.Module):
    def __init__(self, num_FIRs, len_FIRs):
        super(FIRFiltering, self).__init__()
        self.num = num_FIRs  # FIR滤波核个数
        self.len = len_FIRs  # FIR滤波核长度
        wc = torch.linspace(0, pi, self.num+2, requires_grad=True)[1: -1].unsqueeze(dim=1)
        bw = torch.ones([self.num, 1], requires_grad=True)
        self.wc = nn.Parameter(wc)
        self.bw = nn.Parameter(0.01*pi*bw)
        self.filters = Hanning(self.len, self.wc, self.bw)

    def forward(self, waveforms):
        self.filters = Hanning(self.len, self.wc, self.bw).unsqueeze(dim=1).to(waveforms.device)
        self.yout = F.conv1d(waveforms, self.filters, stride=1, padding=int((self.filters.shape[2]-1)/2), dilation=1, bias=None, groups=1)
        return self.yout


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
    def __init__(self, num_FIRs, len_FIRs, num_classes):
        super(BCBLearner, self).__init__()
        # num_FIRs， len_FIRs 是滤波核个数和长度
        self.num_FIRs = num_FIRs
        self.len_FIRs = len_FIRs
        self.FIRFiltering = FIRFiltering(num_FIRs=self.num_FIRs, len_FIRs=self.len_FIRs)
        self.BN = nn.BatchNorm1d(self.num_FIRs)
        # 每个模态提特征，输出长 16 的特征向量
        for ii in range(16):
            exec('self.subNet' + str(ii + 1) + '= SubNet()')
        self.two_class_classifier = nn.Linear(16 * self.num_FIRs, num_classes)

    def forward(self, inputs):
        modes = self.FIRFiltering(inputs)
        out = self.BN(modes)
        for ii in range(self.num_FIRs):
            exec('V' + str(ii+1) + '= self.subNet' + str(ii+1) + '(out[:, ii:ii+1, :])')
        F = torch.tensor([]).to(inputs.device)
        for ii in range(self.num_FIRs):
            F = torch.cat((F, eval('V' + str(ii+1))), dim=1)
        # print(F.shape)
        out = self.two_class_classifier(F)  # [num, 16 * num_FIRs]

        return modes, F, out


class Multi_class_classifier(nn.Module):
    def __init__(self, num_FIRs, num_classes):
        super(Multi_class_classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(16 * num_FIRs * num_classes, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes))

    def forward(self, x):
        return self.fc(x)


class FIR_CCF(nn.Module):
    def __init__(self, num_FIRs, num_classes, *args, **kwargs):
        super(FIR_CCF, self).__init__()
        self.num_classes = num_classes
        # step 1: 多任务学习，每个学习任务基于二值化的伪标签完成跨类滤波与分类
        for ii in range(self.num_classes):
            exec('self.BCBLearner' + str(ii + 1) + '= kwargs[\"BCBLearner' + str(ii + 1) + '\"]')
        # step 2: （参数固定）提取所有任务中的 feature_extractor，然后基于真实标签训练 Multi_class_classifier
        self.classifier = Multi_class_classifier(num_FIRs=num_FIRs, num_classes=num_classes)

    def forward(self, inputs):
        for ii in range(self.num_classes):
            exec('m'+ str(ii+1) + ', F' + str(ii+1) + ', y' + str(ii+1) +  '= self.BCBLearner' + str(ii+1) + '(inputs)')

        V = torch.tensor([]).to(inputs.device)
        for ii in range(self.num_classes):
            V = torch.cat((V, eval('F' + str(ii+1))), dim=1)  # [num, 16 * self.num * num_classes]
        out = self.classifier(V)
        # return m1, m2, m3, m4, m5, F1, F2, F3, F4, F5, y1, y2, y3, y4, y5, out
        return out



if __name__ == '__main__':
    device = torch.device('cuda:1')
    temp = torch.randn([32, 1, 2048]).to(device)
    BCBLearner1 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(device)
    BCBLearner2 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(device)
    BCBLearner3 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(device)
    BCBLearner4 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(device)
    BCBLearner5 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(device)
    model = FIR_CCF(num_FIRs=12, num_classes=5, BCBLearner1=BCBLearner1, BCBLearner2=BCBLearner2,
                    BCBLearner3=BCBLearner3, BCBLearner4=BCBLearner4, BCBLearner5=BCBLearner5).to(device)

    out = model(temp)[-1]
    print(out.shape)

    for name, param in model.named_parameters():
        print(name)



