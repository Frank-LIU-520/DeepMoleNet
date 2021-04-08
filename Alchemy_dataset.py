#!/usr/bin/env python
# encoding: utf-8
# File Name: Alchemy_dataset.py
# Author: Ziteng Liu@ Nanjing University      
# E-mail: njuziteng@hotmail.com
# twitter: MarriotteNJU
# Create Time: 2020/7/08 8:55

import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
import pathlib
import pandas as pd
import numpy as np
import rdkit
from ase.io import read, write
from dscribe.descriptors import SOAP
from dscribe.descriptors import ACSF

Please visit the following website for detail in our codes.

https://itcc.nju.edu.cn/majing/software_en.html
http://106.15.196.160:5659/


