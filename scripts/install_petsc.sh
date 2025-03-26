#!/bin/bash

apt-get update
apt-get install -y build-essential cmake gfortran libblas-dev liblapack-dev libmpich-dev
pip install mpi4py petsc petsc4py