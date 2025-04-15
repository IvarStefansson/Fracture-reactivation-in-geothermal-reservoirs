# Fracture reactivation in geothermal reservoirs Strategies for numerical modeling of fracture contact mechanics in thermoporoelastic media

This repository contains the source code and data for the paper "Fracture reactivation in geothermal reservoirs: Strategies for numerical modeling of fracture contact mechanics in thermoporoelastic media" by  Jakub W. Both, Marius Nevland, Ivar Stefansson, Yury Zabegaev, Eirik Keilegavlen, Inga Berre submitted to the proceedings of The European Geothermal Congress 2025.

## How to develop the code
After cloning the repository, open it in VSCode and run the command Dev Container: Open Workspace in Container, choosing the file devcontainer.code-workspace.
This will open a container with all the necessary dependencies installed, including an editable version of PorePy.

## PorePy version
Use `sh scripts/checkout_porepy.sh` to checkout the commit (f3f14e14a06f5e8245a378502d8c02edebe537a7) used for the development of this repository.

## Extensions of PorePy
Run `sh scripts/install_ncp.sh` to fetch the latest version of the NCP extension of PorePy PorePy and install it.

## Installation of FTHM framework
Run `sh scripts/install_fthm.sh` to fetch the latest version of the FTHM iterative solver framework for PorePy and install it with its dependencies (besides PETSc).

## Installation of PETSc
Run `sh scripts/install_petsc.sh` to install PETSc and the iterative solver framework used in this repository.

## To run the code
To run a simple 3d setup call `python run_simple_bedretto.py`.
The script takes parsed arguments, see the first lines of the script below the if __name__ == "__main__": statement for options and the default values, or call `python run_simple_bedretto.py --help`.
