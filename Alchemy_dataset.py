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

_urls = {
        'dev': 'https://alchemy.tencent.com/data/dev.zip',
        'valid': 'https://alchemy.tencent.com/data/valid.zip',
        'test': 'https://alchemy.tencent.com/data/test.zip',
        }

class TencentAlchemyDataset(InMemoryDataset):   #class could be named QM9, MD17, OR ani-1ccx
    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def __init__(self, root, mode='valid', transform=None, pre_transform=None, pre_filter=None):
        self.mode = mode
        assert mode in _urls
        super(TencentAlchemyDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.mode + '/sdf', self.mode + '/dev_target.csv'] if self.mode == 'dev' else [self.mode + '/sdf', ]

    @property
    def processed_file_names(self):
        return 'TencentAlchemy_%s.pt' % self.mode

    def download(self):
        raise NotImplementedError('please download and unzip dataset from %s, and put it at %s' % (_urls[self.mode], self.raw_dir))

    def alchemy_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            h_t += [int(d['chiralty'] == x) \
                       for x in (rdkit.Chem.rdchem.ChiralType.CHI_OTHER, 
                       rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                       rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                       rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED)]
            h_t.extend(d['chiralrs'])
            h_t.append(int(d['formal_charge']))
            h_t.append(1 if d['inring'] == 'True' else 0)
            h_t.append(1 if d['inring3'] == 'True' else 0)
            h_t.append(1 if d['inring4'] == 'True' else 0)
            h_t.append(1 if d['inring5'] == 'True' else 0)
            h_t.append(1 if d['inring6'] == 'True' else 0)
            h_t.append(1 if d['inring7'] == 'True' else 0)
            h_t.append(1 if d['inring8'] == 'True' else 0)
            h_t.append(1 if d['inring9'] == 'True' else 0)
            h_t.append(1 if d['inring10'] == 'True' else 0)
            h_t.append(1 if d['inring11'] == 'True' else 0)
            h_t.append(d['partialcharge'])
            h_t.append(d['Rvdw'])
            h_t.append(d['Defaultvalence'])
            h_t.append(d['Oterelecs'])
            h_t.append(d['Rb0'])      
            h_t.append(int(d['totaldegree']))
            h_t.append(d['envr1'])
            h_t.append(d['envr2'])
            h_t.append(d['envr3'])
            h_t.append(d['envr4'])
            h_t.append(d['envr5'])
            h_t.append(d['envr6'])
            h_t.append(d['envr7'])
            h_t.append(d['envr8'])
            h_t.extend(d['env4ascf'])

              
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def alchemy_edges(self, g):
        e={}
        for n1, n2, d in g.edges(data=True):
            if d['b_type']  != 'nobond':
              e_t = [int(d['b_type'] == x)
                    for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]
            else:
              e_t= [0 , 0, 0, 0]
            e_t.append(d['shortestpath'])
            e_t.append(d['shortestbonds'])
            e_t.append(d['insamering'])
            e_t.extend(d['graph_dis'])
            e_t.extend(d['extend_dis'])
            e[(n1, n2)] = e_t
        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def sdf_graph_reader(self, sdf_file):
        atom4ascf = read(sdf_file)
        with open(sdf_file, 'r') as f:
            sdf_string = f.read()
        mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)
        AllChem.ComputeGasteigerCharges(mol)
        peri=Chem.rdchem.GetPeriodicTable()
        if mol is None:
            print("rdkit can not parsing", sdf_file)
            return None
        feats = self.chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()
        l = torch.FloatTensor(self.target.loc[int(sdf_file.stem)].tolist()).unsqueeze(0) \
                if self.mode == 'dev' else torch.LongTensor([int(sdf_file.stem)])
        id= torch.LongTensor([int(sdf_file.stem)])
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()
        for i in range(mol.GetNumAtoms()):
         atom_i = mol.GetAtomWithIdx(i)
         
         g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                num_h=atom_i.GetTotalNumHs(),chiralty=atom_i.GetChiralTag(),chiralrs=[0,0],
                formal_charge=atom_i.GetFormalCharge(), inring=atom_i.IsInRing(),
                inring3=atom_i.IsInRingSize(3),inring4=atom_i.IsInRingSize(4),inring5=atom_i.IsInRingSize(5),inring6=atom_i.IsInRingSize(6),
                inring7=atom_i.IsInRingSize(7),inring8=atom_i.IsInRingSize(8),inring9=atom_i.IsInRingSize(9),inring10=atom_i.IsInRingSize(10),
                inring11=atom_i.IsInRingSize(11),partialcharge=float(atom_i.GetProp('_GasteigerCharge')), 
                Rvdw=peri.GetRvdw(atom_i.GetAtomicNum()) ,Defaultvalence=peri.GetDefaultValence(atom_i.GetAtomicNum()),
                Oterelecs=peri.GetNOuterElecs(atom_i.GetAtomicNum()) ,Rb0=peri.GetRb0(atom_i.GetAtomicNum()),
                    totaldegree=atom_i.GetTotalDegree(),envr1=self.envatom(mol,1,atom_i.GetIdx()),
                    envr2=self.envatom(mol,2,atom_i.GetIdx()),envr3=self.envatom(mol,3,atom_i.GetIdx()),
                   envr4=self.envatom(mol,4,atom_i.GetIdx()),envr5=self.envatom(mol,5,atom_i.GetIdx()),envr6=self.envatom(mol,6,atom_i.GetIdx()),
                   envr7=self.envatom(mol,7,atom_i.GetIdx()),envr8=self.envatom(mol,8,atom_i.GetIdx()) , env4ascf=self.envascf(atom4ascf,atom_i.GetIdx()))
 
        
        for i in range(len(feats)):
         if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                g.node[i]['donor'] = 1
         elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                g.node[i]['acceptor'] = 1
        rs=Chem.FindMolChiralCenters(mol)
        for i in rs:
         if i[1] == 'R':
          g.node[i[0]]['chiralrs'] = [1,0]
         elif i[1] == 'S':
          g.node[i[0]]['chiralrs'] = [0,1]
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j, b_type=e_ij.GetBondType())
                else:
                    g.add_edge(i, j, b_type='nobond')
                
                if i != j  :
                    g.add_edge(i,j,shortestpath= len(Chem.rdmolops.GetShortestPath(mol, i, j))-1    )
                else:
                    g.add_edge(i,j,shortestpath= 0    )
                    
                if i != j  :
                    g.add_edge(i,j,shortestbonds= self.shortestpathtotbonds(mol,i,j)    )
                else:
                    g.add_edge(i,j,shortestbonds= 0    )
                    
                if i != j  :
                    g.add_edge(i,j,insamering= self.samering(mol,i,j)    )
                else:
                    g.add_edge(i,j,insamering= 0    )                
    
                
                if i!=j:
                    tmp=[0]*13
                    shortestpath=len(Chem.rdmolops.GetShortestPath(mol, i, j))-1
                    if  (len(Chem.rdmolops.GetShortestPath(mol, i, j))-1) == self.shortestpathtotbonds(mol,i,j) :
                        tmp[shortestpath-1]=1
                        g.add_edge(i,j,graph_dis=tmp)
                    else:
                        g.add_edge(i,j,graph_dis=[0]*13)
                else:
                    g.add_edge(i,j,graph_dis=[0]*13)
                    
                if i !=j :
                    d_i=mol.GetConformers()[0].GetPositions()[i]
                    d_j=mol.GetConformers()[0].GetPositions()[j]
                    ext_dis=np.linalg.norm(d_i-d_j)
                    
                    g.add_edge(i,j,extend_dis=np.exp(-(ext_dis-np.linspace(0,4,20))**2/(0.5**2)))
                    
                else:
                    g.add_edge(i,j,extend_dis=[0]*20)
        node_attr = self.alchemy_nodes(g)
        edge_index, edge_attr = self.alchemy_edges(g)
        data = Data(
                x=node_attr,
                pos=torch.FloatTensor(geom),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=l,
				id=id
                )
        return data

    def shortestpathtotbonds(self,m,i,j):
        path_atom_list=Chem.rdmolops.GetShortestPath(m, i, j)
        path_tot_bonds=0
        for  i in range(len(path_atom_list)-1):
          path_tot_bonds=path_tot_bonds+m.GetBondBetweenAtoms(path_atom_list[i],path_atom_list[i+1]).GetBondTypeAsDouble()
        return  path_tot_bonds

    def samering(self,m,i,j):
        ri=m.GetRingInfo()
        ringcount=[]
        for ring in range(m.GetRingInfo().NumRings()):
            if i in ri.AtomRings()[ring]  and  j in ri.AtomRings()[ring]:
                ringcount.append(1)
            else:
                ringcount.append(0)
        if 1 in ringcount:
            out=1
        else:
            out=0
        return out

    def envatom(self,mol,j,i):  
       env= Chem.FindAtomEnvironmentOfRadiusN(mol,j,i,useHs=True)
       amap={}
       submol=Chem.PathToSubmol(mol,env,atomMap=amap)
       return submol.GetNumAtoms()
    
    def envascf(self,mol4ascf,position):
       acsf = ACSF(
       species = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl'],
       rcut=6.0,
       g2_params=[[1, 1], [1, 2], [1, 3]],
       g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
       acsf4atom = acsf.create(mol4ascf, positions=[position])
       return acsf4atom[0]




    def process(self):
        if self.mode == 'dev' or 'valid':
            self.target = pd.read_csv(self.raw_paths[1], index_col=0,
                    usecols=['gdb_idx',] + ['property_%d' % x for x in range(12)])
            self.target = self.target[['property_%d' % x for x in range(12)]]
        sdf_dir = pathlib.Path(self.raw_paths[0])
        data_list = []
        for sdf_file in sdf_dir.glob("**/*.sdf"):
            alchemy_data = self.sdf_graph_reader(sdf_file)
            if alchemy_data is not None:
                data_list.append(alchemy_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


