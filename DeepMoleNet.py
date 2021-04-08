#!/usr/bin/env python
# encoding: utf-8
# File Name: DeepMoleNet.py
# Author: Ziteng Liu@ Nanjing University      
# E-mail: njuziteng@hotmail.com
# twitter: MarriotteNJU
# Create Time: 2020/7/08 8:55

import math
import torch_geometric.transforms as T
from Alchemy_dataset import TencentAlchemyDataset
from torch_geometric.nn import NNConv, Set2Set,global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops,softmax
from torch.optim import lr_scheduler
import os
import pandas as pd
import numpy as np
import json
from ase.units import Hartree, eV, Bohr, Ang
import torch
import torch.nn as nn
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler



Please visit the following website for detail in our codes.

https://itcc.nju.edu.cn/majing/software_en.html
http://106.15.196.160:5659/


