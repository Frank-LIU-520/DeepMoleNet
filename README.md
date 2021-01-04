# DeepMoleNet

After download the code, one can use DeepMoleNet model in the following steps:
    1. To unzip the download zip file and make sure that all packages (including ase, networkx, torch, torch-cluster, torch-geometric, torch-scatter, rdkit, pathlib, pandas, numpy, dscribe) listed in the requirement.txt file are installed.
    2. To put molecule sdf files(QM9, MD17, ANI-1ccx and etc) in the .\data-bin\raw\dev, with its sdf file name and all 12 properties saved in .\data-bin\raw\dev_target.csv in templet.
    3. To run the code, python DeepMoleNet.py


We also provide trained models of single-target and multi-target results trained on QM9, models trained on MD17 and models trained on ANI-1ccx for reporoducibility and other uses like transfer learning on small dataset.

To cite this algorithm, please reference: Liu, Ziteng, et al. "Transferable multi-level attention neural network for accurate prediction of quantum chemistry properties via multi-task learning." ChemRxiv 12588170 (2020): v1.

Please feel free to contact us via njuziteng@hotmail.com for any questions.
