#!/bin/bash
# Install the FTHM library

# Set up environment variables
FTHM_DIR=$HOME/fthm

cd $HOME

if [ ! -d "$FTHM_DIR" ]; then
  git clone https://github.com/pmgbergen/FTHM-Solver.git $FTHM_DIR
  cd $FTHM_DIR
else
  cd $FTHM_DIR
  git pull
fi

pip install -e .
