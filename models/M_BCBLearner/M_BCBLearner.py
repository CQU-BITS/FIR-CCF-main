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
from models.FIR_CCF.FIR_CCF import BCBLearner


class M_BCBLearner(nn.Module):
    def __init__(self, num_FIRs, len_FIRs, num_classes):
        super(M_BCBLearner, self).__init__()
        # num_FIRs， len_FIRs 是滤波核个数和长度
        self.num_FIRs = num_FIRs
        self.len_FIRs = len_FIRs

        self.net = BCBLearner(num_FIRs=num_FIRs, len_FIRs=len_FIRs, num_classes=num_classes)

    def forward(self, inputs):
        m, F, out = self.net(inputs)
        return out


if __name__ == '__main__':
    device = torch.device('cuda:1')
    temp = torch.randn([32, 1, 2048]).to(device)
    model = M_BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=5).to(device)
    F, out = model(temp)
    print(out.shape)
    # for name, param in SubModule.named_parameters():
    #     print(name)

