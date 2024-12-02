# Authors   : Rui Liu, Xiaoxi Ding, Shenglan Liu, Hebin Zheng, Yuanyaun Xu, Yimin Shao
# URL       : https://www.sciencedirect.com/science/article/pii/S0951832024006811
# Reference : R. Liu, X. Ding, S. Liu, H. Zheng, Y. Xu, Y. Shao, Knowledge-informed FIR-based cross-category filtering framework for
#             interpretable machinery fault diagnosis under small samples, Reliab. Eng. Syst. Saf., 254 (2025) 110610.
# DOI       : https://doi.org/10.1016/j.ress.2024.110610
# Date      : 2024/12/02
# Version   : V0.1.0
# Copyright by CQU-BITS


import os, random
import warnings
import torch
import time
import numpy as np
import datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from postprocessing.myplots import plotLossAcc, plotTSNECluster, plotConfusionMatrix
import models
from models.FIR_CCF.FIR_CCF import BCBLearner


speed = 400
shot = 1
cond = str(speed) + '_' + str(shot)
batch_size = 128
lr = 1e-3
max_epochs = 100
model_name = 'FIR_CCF'
dataset_name = 'CQU_gear_dataset'  # choices=['CQU_gear_dataset', 'SEU_bearing_dataset', 'MCC5_THU_gearbox_dataset']
result_file_save_path = os.path.join('Results', dataset_name, model_name, cond)


class train_utils(object):
    def setup(self):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.dataset_name = dataset_name
        self.model_name = model_name
        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_count = torch.cuda.device_count()
            print('using {} gpus'.format(self.device_count))
            assert batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            print('using {} cpu'.format(self.device_count))
        # Load the datasets
        self.datasets = {}
        Dataset = getattr(datasets, dataset_name)
        if dataset_name in ['CQU_gear_dataset', 'SEU_bearing_dataset']:
            self.datasets['fake_train1'], self.datasets['fake_train2'], self.datasets['fake_train3'], self.datasets['fake_train4'], \
            self.datasets['fake_train5'], self.datasets['real_train'], self.datasets['real_test'] \
                = Dataset(speed=speed,
                          snr=None,
                          normlizetype='mean~std',
                          shot=shot).data_preprare()
            self.dataloaders = {x: DataLoader(self.datasets[x],
                                              batch_size=(self.batch_size if 'train' in x else 1000),
                                              shuffle=(True if 'train' in x else False),
                                              pin_memory=(True if self.device == 'cuda' else False))
                                for x in ['fake_train1', 'fake_train2', 'fake_train3', 'fake_train4', 'fake_train5',
                                          'real_train', 'real_test']}
        elif dataset_name in ['MCC5_THU_gearbox_dataset']:
            self.datasets['fake_train1'], self.datasets['fake_train2'], self.datasets['fake_train3'], self.datasets['fake_train4'], \
            self.datasets['fake_train5'], self.datasets['fake_train6'], self.datasets['fake_train7'], self.datasets['fake_train8'], \
            self.datasets['real_train'], self.datasets['real_test'] \
                = Dataset(speed=speed,
                          snr=None,
                          normlizetype='mean~std',
                          shot=shot).data_preprare()
            self.dataloaders = {x: DataLoader(self.datasets[x],
                                              batch_size=(self.batch_size if 'train' in x else 1000),
                                              shuffle=(True if 'train' in x else False),
                                              pin_memory=(True if self.device == 'cuda' else False))
                                for x in ['fake_train1', 'fake_train2', 'fake_train3', 'fake_train4', 'fake_train5',
                                          'fake_train6', 'fake_train7', 'fake_train8', 'real_train', 'real_test']}
        else:
            assert "The dataset is not involved"
        # Load the model to device

        if self.model_name != 'Conv_CCF':
            assert "Selected model is not involved"

        # step 1: 建立SubModule，多任务学习，每个学习任务基于二元的伪标签完成跨类滤波与分类
        if dataset_name in ['CQU_gear_dataset', 'SEU_bearing_dataset']:
            BCBLearner1 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner2 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner3 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner4 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner5 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner1 = self.pre_train_BCBLearner(BCBLearner=BCBLearner1, fake_trainSet=self.dataloaders['fake_train1'])
            BCBLearner2 = self.pre_train_BCBLearner(BCBLearner=BCBLearner2, fake_trainSet=self.dataloaders['fake_train2'])
            BCBLearner3 = self.pre_train_BCBLearner(BCBLearner=BCBLearner3, fake_trainSet=self.dataloaders['fake_train3'])
            BCBLearner4 = self.pre_train_BCBLearner(BCBLearner=BCBLearner4, fake_trainSet=self.dataloaders['fake_train4'])
            BCBLearner5 = self.pre_train_BCBLearner(BCBLearner=BCBLearner5, fake_trainSet=self.dataloaders['fake_train5'])
            self.model = getattr(models, self.model_name)(num_FIRs=12, num_classes=Dataset.num_classes,
                                                          BCBLearner1=BCBLearner1, BCBLearner2=BCBLearner2,
                                                          BCBLearner3=BCBLearner3, BCBLearner4=BCBLearner4,
                                                          BCBLearner5=BCBLearner5).to(self.device)
        else:
            BCBLearner1 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner2 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner3 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner4 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner5 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner6 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner7 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner8 = BCBLearner(num_FIRs=12, len_FIRs=193, num_classes=2).to(self.device)
            BCBLearner1 = self.pre_train_BCBLearner(BCBLearner=BCBLearner1, fake_trainSet=self.dataloaders['fake_train1'])
            BCBLearner2 = self.pre_train_BCBLearner(BCBLearner=BCBLearner2, fake_trainSet=self.dataloaders['fake_train2'])
            BCBLearner3 = self.pre_train_BCBLearner(BCBLearner=BCBLearner3, fake_trainSet=self.dataloaders['fake_train3'])
            BCBLearner4 = self.pre_train_BCBLearner(BCBLearner=BCBLearner4, fake_trainSet=self.dataloaders['fake_train4'])
            BCBLearner5 = self.pre_train_BCBLearner(BCBLearner=BCBLearner5, fake_trainSet=self.dataloaders['fake_train5'])
            BCBLearner6 = self.pre_train_BCBLearner(BCBLearner=BCBLearner6, fake_trainSet=self.dataloaders['fake_train6'])
            BCBLearner7 = self.pre_train_BCBLearner(BCBLearner=BCBLearner7, fake_trainSet=self.dataloaders['fake_train7'])
            BCBLearner8 = self.pre_train_BCBLearner(BCBLearner=BCBLearner8, fake_trainSet=self.dataloaders['fake_train8'])
            self.model = getattr(models, self.model_name)(num_FIRs=12, num_classes=Dataset.num_classes,
                                                          BCBLearner1=BCBLearner1, BCBLearner2=BCBLearner2,
                                                          BCBLearner3=BCBLearner3, BCBLearner4=BCBLearner4,
                                                          BCBLearner5=BCBLearner5, BCBLearner6=BCBLearner6,
                                                          BCBLearner7=BCBLearner7, BCBLearner8=BCBLearner8).to(self.device)

        # 定义 loss 评判准则和优化器, 只更新 global classifier 的参数
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        classifier_list = (param for name, param in self.model.named_parameters() if 'classifier' in name)
        self.optimizer = optim.Adam([{'params': classifier_list, 'lr': 1e-3}])


    def pre_train_BCBLearner(self, BCBLearner, fake_trainSet):
        # Define the loss type and optimizer for SubFilterNet1
        criterion = nn.CrossEntropyLoss()
        wc_list = (param for name, param in BCBLearner.named_parameters() if 'FIRFiltering.wc' in name)
        bw_list = (param for name, param in BCBLearner.named_parameters() if 'FIRFiltering.bw' in name)
        others_list = (param for name, param in BCBLearner.named_parameters() if 'FIRFiltering' not in name)
        optimizer = optim.Adam([{'params': wc_list, 'lr': 5e-3},
                                {'params': bw_list, 'lr': 1e-4},
                                {'params': others_list, 'lr': 1e-3}])
        # 子模型训练：跨类滤波
        for epoch in range(300):
            BCBLearner.train()
            epoch_acc = 0
            epoch_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(fake_trainSet):
                inputs, labels = inputs.to(self.device), labels.to(self.device, dtype=torch.long)
                _, F, logits = BCBLearner(inputs)
                ce_loss = criterion(logits, labels)
                l1_loss = torch.abs(F).sum()
                loss = ce_loss + 1e-5 * l1_loss
                loss_temp = loss.item() * inputs.size(0)
                epoch_loss += loss_temp
                epoch_acc += torch.eq(logits.argmax(dim=1), labels).float().sum().item()

                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 梯度计算与反向传播
                optimizer.step()  # 参数更新
            epoch_train_loss = epoch_loss / len(fake_trainSet.dataset)
            epoch_train_acc = epoch_acc / len(fake_trainSet.dataset)
            print('{} epoch, BCBLearner 预训练Loss: {:.8f}, 预练精度: {:.4%}  >> [{}/{}]'.format(
                epoch, epoch_train_loss, epoch_train_acc, int(epoch_acc), len(fake_trainSet.dataset)))
        return BCBLearner


    def my_iterator(self):  # Training and testing process
        start_time = time.time()
        best_train_acc = 0.0
        best_test_acc = 0.0
        trainLoss, trainAcc, testLoss, testAcc = [], [], [], []
        for epoch in range(self.max_epochs):
            LOG, LAB = torch.tensor([]), torch.tensor([])
            for phase in ['real_train', 'real_test']:
                epoch_loss = 0.0
                epoch_acc = 0
                if phase == 'real_train':
                    self.model.train()
                else:
                    self.model.eval()
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs, labels = inputs.to(self.device), labels.to(self.device, dtype=torch.long)
                    with torch.set_grad_enabled(phase == 'real_train'):
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct_num = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)  # criterion返回的 loss 是每个 batch 所有样本的平均值
                        epoch_loss += loss_temp
                        epoch_acc += correct_num

                        # 训练集,模型更新
                        if phase == 'real_train':
                            self.optimizer.zero_grad()  # 梯度清零
                            loss.backward()  # 梯度计算与反向传播
                            self.optimizer.step()  # 参数更新
                        else:  # phase == 'real_test', 测试集特征，用于生产聚类图和混淆矩阵
                            LOG = torch.cat((LOG, logits.cpu()), dim=0)
                            LAB = torch.cat((LAB, labels.unsqueeze(dim=-1).cpu()), dim=0)

                if phase == 'real_train':
                    epoch_train_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                    epoch_train_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                    trainLoss.append(epoch_train_loss)
                    trainAcc.append(epoch_train_acc)
                    print('{} epoch, 训练Loss: {:.8f}, 训练精度: {:.4%}  >> [{}/{}]'.format(
                        epoch, epoch_train_loss, epoch_train_acc, int(epoch_acc), len(self.dataloaders[phase].dataset)))
                else:  # phase == 'real_test':
                    epoch_test_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                    epoch_test_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                    testLoss.append(epoch_test_loss)
                    testAcc.append(epoch_test_acc)
                    print('{} epoch, 测试Loss: {:.8f}, 测试精度: {:.4%}  >> [{}/{}]'.format(
                        epoch, epoch_test_loss, epoch_test_acc, int(epoch_acc), len(self.dataloaders[phase].dataset)))

                    # 基于训练结果计算最佳测试精度与训练时间
                    if epoch_train_acc >= best_train_acc:
                        best_train_acc = epoch_train_acc
                        best_test_acc = epoch_test_acc
                        train_time = time.time() - start_time
        print('---------------最终测试精度为: {:.4%}---------------------'.format(best_test_acc))
        return LOG.numpy(), LAB.numpy(), trainLoss, trainAcc, testLoss, testAcc, best_test_acc, train_time


if __name__ == '__main__':
    trainer = train_utils()
    trainer.setup()
    LOG, LAB, trainLoss, trainAcc, testLoss, testAcc, best_test_acc, train_time = trainer.my_iterator()
    if dataset_name == 'MCC5_THU_gearbox_dataset':
        typeName = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    else:
        typeName = ['C1', 'C2', 'C3', 'C4', 'C5']
    plotLossAcc(trainLoss, testLoss, trainAcc, testAcc, result_file_save_path, '收敛曲线')
    plotConfusionMatrix(torch.from_numpy(LOG).argmax(dim=-1), LAB, typeName, result_file_save_path, '混淆矩阵')
    plotTSNECluster(np.hstack([LOG, LAB]), len(typeName), typeName, result_file_save_path, '聚类图')

    if os.path.exists(result_file_save_path) == False:
        os.makedirs(result_file_save_path)
    np.savetxt(result_file_save_path + '/' + 'epoch-train_loss曲线.csv', trainLoss, fmt="%.8f")
    np.savetxt(result_file_save_path + '/' + 'epoch-train_acc曲线.csv', testAcc, fmt="%.8f")

    try_times = 10
    try_test_Acc = []
    try_train_Time = []
    for ii in range(try_times):
        trainer = train_utils()
        trainer.setup()
        _, _, _, _, _, testAcc, best_test_acc, train_time = trainer.my_iterator()
        try_test_Acc.append(best_test_acc)
        try_train_Time.append(train_time)
    mean_test_acc = np.mean(try_test_Acc)
    std_test_acc = np.std(try_test_Acc)
    try_test_Acc.append(mean_test_acc)
    try_test_Acc.append(std_test_acc)
    np.savetxt(result_file_save_path + '/' + 'try_test_Acc.csv', try_test_Acc, fmt="%.8f")
    print('循环', try_times, '次，平均测试精度：{:.4%}, 标注差：{:.4%}'.format(mean_test_acc, std_test_acc))


