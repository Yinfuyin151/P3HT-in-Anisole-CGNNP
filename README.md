# P3HT in Anisole CGNNP

This repository contains the data and code supporting the P3HT coarse-graining
study in anisole. It includes molecular dynamics input files, the trained model
and force-field files, representative trajectories, and the data used to
generate the figures.

## Repository Contents

- `Analysis/`: analysis scripts and processed data for structural and dynamical
  comparisons, including bond/angle distributions, end-to-end distances, RDF,
  radius of gyration, persistence length, MSD, and cluster-related results.
- `MDdemo/`: demonstration input files for running representative all-atom,
  CGNNP coarse-grained, and Martini 3 molecular dynamics simulations.
- `Models/`: trained CGNNP model checkpoints, training configuration files,
  training metrics, and scripts for inspecting loss and force predictions.
- `Preprocessing/`: mapping and preprocessing files used to construct the
  coarse-grained representation and residual-force targets.
- `Trajectories/`: representative single-chain and multichain trajectories and
  visualization files used for comparison and figure generation.

The molecular dynamics and machine-learning workflows are based on
[TorchMD](https://github.com/torchmd/torchmd) and
[TorchMD-CG](https://github.com/torchmd/torchmd-cg). The residual-force target
construction was adapted from TorchMD-CG and modified for the P3HT
coarse-graining framework used in this work.

Large trajectory files (`*.xtc`) are stored using Git LFS.
