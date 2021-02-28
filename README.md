# DeepMoleNet


  DeepMoleNet is a deep learning package for molecular properties prediction. It is developed with a multi-level attention mechanism to enable chemical interpretable insights being fused into multi-task learning through (1) weighting contributions from various atoms and (2) taking the atom-centered symmetry functions (ACSFs) as the teacher descriptor, rather than using ACSFs as input in the conventional way. The properties including dipole moment, HOMO, and Gibbs free energy within chemical accuracy are achieved by using multiple benchmarks, both at the equilibrium and non-equilibrium geometries.

      To cite this algorithm, please reference: Ziteng Liu, Liqiang Lin, Qingqing Jia, Zheng Cheng, Yanyan Jiang, Yanwen Guo*, Jing Ma*. "Transferable multilevel attention neural network for accurate prediction of quantum chemistry properties via multitask learning." JCIM 2021 https://doi.org/10.1021/acs.jcim.0c01224   https://pubs.acs.org/doi/10.1021/acs.jcim.0c01224#
      
After download the code, one can use DeepMoleNet model in the following steps:

    1. To unzip the download zip file and make sure that all packages (including ase, networkx, torch, torch-cluster, torch-geometric, torch-scatter, rdkit, pathlib, pandas, numpy, dscribe) listed in the requirement.txt file are installed.
    
    2. To put molecule sdf files(QM9, Alchemy, MD17, ANI-1ccx and etc) in the .\data-bin\raw\dev, with its sdf file name and all 12 properties saved in .\data-bin\raw\dev_target.csv in templet.
    
    3. To run the code, python DeepMoleNet.py


We also provide trained models of single-target and multi-target results trained on QM9, models trained on MD17 and models trained on ANI-1ccx for reporoducibility and other uses like transfer learning on small dataset.

Please feel free to contact us via majing@nju.edu.cn for any questions.
