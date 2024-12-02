#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch

from models.DRSN.DRSN import rsnet18 as DRSN
from models.LiConvFormer.LiConvFormer import LiConvFormer
from models.MTAGN.MTAGN import MTAGN
from models.EWSNet.EWSNet import Net as EWSNet
from models.M_BCBLearner.M_BCBLearner import M_BCBLearner
from models.Conv_CCF.Conv_CCF import Conv_CCF
from models.FIR_CCF.FIR_CCF import FIR_CCF
from models.LWK_CCF.LWK_CCF import LWK_CCF