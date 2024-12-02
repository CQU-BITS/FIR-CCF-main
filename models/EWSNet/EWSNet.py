# Reference: C. He, H. Shi, J. Si, J. Li, Physics-informed interpretable wavelet weight initialization and balanced dynamic
#            adaptive threshold for intelligent fault diagnosis of rolling bearings, J. Manuf. Syst., 70 (2023) 579-592.


import torch
import torch.nn as nn
# from wmodelsii8 import Sin_fast as fast
# from wmodelsii3 import Laplace_fast as fast
from models.EWSNet.weights import Laplace_fastv2 as fast
# from wsinc import SincConv_fast as fast
from models.EWSNet.thresholds import Shrinkagev3ppp2 as sage


class Mish1(nn.Module):
    def __init__(self):
        super(Mish1, self).__init__()
        self.mish = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.mish(x)


class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()    #85,42,70   #63,31,75
        self.p1_0 = nn.Sequential(  # nn.Conv1d(1, 50, kernel_size=18, stride=2),
            # fast(out_channels=64, kernel_size=250, stride=1),
            # fast1(out_channels=70, kernel_size=84, stride=1),
            nn.Conv1d(in_channels, 64, kernel_size=250, stride=1, bias=True),
            nn.BatchNorm1d(64),
            Mish1()
        )
        self.p1_1 = nn.Sequential(nn.Conv1d(64, 16, kernel_size=18, stride=2, bias=True),
                                  # fast(out_channels=50, kernel_size=18, stride=2),
                                  nn.BatchNorm1d(16),
                                  Mish1()
                                  )
        self.p1_2 = nn.Sequential(nn.Conv1d(16, 10, kernel_size=10, stride=2, bias=True),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                  )
        self.p1_3 = nn.MaxPool1d(kernel_size=2)
        self.p2_1 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=6, stride=1, bias=True),
                                  # fast(out_channels=50, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(32),
                                  Mish1()
                                  )
        self.p2_2 = nn.Sequential(nn.Conv1d(32, 16, kernel_size=6, stride=1, bias=True),
                                  nn.BatchNorm1d(16),
                                  Mish1()
                                  )
        self.p2_3 = nn.MaxPool1d(kernel_size=2)
        self.p2_4 = nn.Sequential(nn.Conv1d(16, 10, kernel_size=6, stride=1, bias=True),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                  )
        self.p2_5 = nn.Sequential(nn.Conv1d(10, 10, kernel_size=8, stride=2, bias=True),
                                  # nn.Conv1d(10, 10, kernel_size=6, stride=2),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                 )  # PRelu
        self.p2_6 = nn.MaxPool1d(kernel_size=2)
        self.p3_0 = sage(channel=64, gap_size=1)
        self.p3_1 = nn.Sequential(nn.Conv1d(64, 10, kernel_size=43, stride=4, bias=True),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                 )
        self.p3_2 = nn.MaxPool1d(kernel_size=2)
        self.p3_3 = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.p4 = nn.Sequential(nn.Linear(10, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.kernel_size == (500,):
                    m.weight.data = fast(out_channels=64, kernel_size=250).forward()
                    nn.init.constant_(m.bias.data, 0.0)
                else:
                    nn.init.kaiming_normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(1)

    def forward(self, x):
        x = self.p1_0(x)
        p1 = self.p1_3(self.p1_2(self.p1_1(x)))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))
        x = self.p3_2(self.p3_1(x + self.p3_0(x)))
        x = torch.add(x, torch.add(p1, p2))
        x = self.p3_3(x).squeeze()
        x = self.p4(x) 
        return x

if __name__ == '__main__':

    input = torch.randn(32, 1, 2048).cuda()
    model = Net(in_channels=1, num_classes=5).cuda()
    # for param in model.parameters():
    #     print(type(param.data), param.size())
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    output = model(input)
    print(output.shape)


